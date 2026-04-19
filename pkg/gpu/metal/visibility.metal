// visibility.metal
// Metal Performance Shaders for Sensor Visibility Computation
// ADITI 4.0: 100k objects × M sensors in milliseconds on M2

#include <metal_stdlib>
using namespace metal;

// Earth parameters (WGS84)
constant float EARTH_RADIUS_KM = 6378.137;
constant float EARTH_OBLATENESS = 1.0 / 298.257223563; // 1/f

// Visibility result structure
struct VisibilityResult {
    uint32_t objectID;      // Object index
    uint32_t sensorID;      // Sensor index  
    bool isVisible;         // Line-of-sight clear?
    float range_km;         // Distance to object
    float elevation_deg;    // Above horizon
    float phase_angle_deg;  // Sun-target-sensor angle
};

// Object state (position only for visibility)
struct ObjectState {
    float3 position;      // ECI position (km)
    float3 velocity;      // ECI velocity (km/s) - for doppler
    uint32_t objectID;
    float radius_m;       // Object size (for detection threshold)
};

// Sensor platform state
struct SensorState {
    float3 position;        // ECI position (km) - satellite or ground
    float3 boresight;       // Pointing direction (unit vector)
    float fov_halfangle;    // Field of view (radians)
    float range_limit_km;   // Maximum detection range
    float min_elevation;    // For ground sensors (radians)
    uint32_t sensorID;
    uint32_t sensorType;    // 0=optical, 1=radar, 2=rf
};

// Ray-sphere intersection for Earth occlusion
// Returns true if ray from origin to target clears Earth
bool lineOfSightClear(float3 origin, float3 target, 
                      float earthRadius, float oblateness) {
    float3 ray = target - origin;
    float rayLength = length(ray);
    
    // Normalize
    float3 direction = ray / rayLength;
    
    // Earth center to ray origin
    float3 toOrigin = -origin; // Vector from origin to Earth center
    
    // Project onto ray direction
    float projection = dot(toOrigin, direction);
    
    // Closest approach to Earth center
    float3 closestPoint = origin + direction * projection;
    float closestDist = length(closestPoint);
    
    // Account for Earth oblateness at this latitude
    float latitude = asin(closestPoint.z / closestDist);
    float localRadius = earthRadius * (1.0 - oblateness * sin(latitude) * sin(latitude));
    
    // Check if ray passes through Earth
    if (closestDist < localRadius) {
        // Ray intersects Earth - check if intersection is between origin and target
        float halfChord = sqrt(localRadius * localRadius - closestDist * closestDist);
        float entryPoint = projection - halfChord;
        float exitPoint = projection + halfChord;
        
        // If entry point is positive and less than ray length, it's occluded
        return !(entryPoint > 0.0 && entryPoint < rayLength);
    }
    
    return true;
}

// Compute visibility metrics for object-sensor pair
VisibilityResult computeVisibility(ObjectState obj, SensorState sensor, 
                                   float3 sunDirection) {
    VisibilityResult result;
    result.objectID = obj.objectID;
    result.sensorID = sensor.sensorID;
    result.isVisible = false;
    result.range_km = 0.0;
    result.elevation_deg = -90.0;
    result.phase_angle_deg = 0.0;
    
    // Vector from sensor to object
    float3 sensorToObj = obj.position - sensor.position;
    result.range_km = length(sensorToObj);
    
    // Check range limit
    if (result.range_km > sensor.range_limit_km) {
        return result;
    }
    
    // Check Earth occlusion (line of sight)
    if (!lineOfSightClear(sensor.position, obj.position, 
                          EARTH_RADIUS_KM, EARTH_OBLATENESS)) {
        return result; // Behind Earth
    }
    
    // Compute elevation (for ground sensors)
    float3 sensorUp = normalize(sensor.position); // For ground stations
    float3 objDirection = normalize(sensorToObj);
    float cosElevation = dot(objDirection, sensorUp);
    result.elevation_deg = degrees(acos(cosElevation));
    
    if (result.elevation_deg < degrees(sensor.min_elevation)) {
        return result; // Below horizon
    }
    
    // Check FOV (for space-based sensors)
    float cosAngle = dot(normalize(sensorToObj), sensor.boresight);
    if (cosAngle < cos(sensor.fov_halfangle)) {
        return result; // Outside FOV
    }
    
    // Compute phase angle (Sun-target-sensor)
    float3 toSun = normalize(sunDirection);
    float3 toSensor = -objDirection; // From object to sensor
    float cosPhase = dot(toSun, toSensor);
    result.phase_angle_deg = degrees(acos(cosPhase));
    
    // All checks passed
    result.isVisible = true;
    
    return result;
}

// Kernel: Visibility check for all object-sensor pairs
// Grid: (numObjects, numSensors) threads
kernel void visibilityCheckKernel(
    device const ObjectState* objects [[buffer(0)]],
    device const SensorState* sensors [[buffer(1)]],
    device VisibilityResult* results [[buffer(2)]],
    constant uint& numObjects [[buffer(3)]],
    constant uint& numSensors [[buffer(4)]],
    constant float3& sunDirection [[buffer(5)]],
    uint2 gridPosition [[thread_position_in_grid]]) {
    
    uint objIdx = gridPosition.x;
    uint sensorIdx = gridPosition.y;
    
    // Bounds check
    if (objIdx >= numObjects || sensorIdx >= numSensors) {
        return;
    }
    
    // Compute visibility
    uint resultIdx = objIdx * numSensors + sensorIdx;
    results[resultIdx] = computeVisibility(
        objects[objIdx], 
        sensors[sensorIdx],
        sunDirection
    );
}

// Kernel: Filter visible pairs only (parallel stream compaction)
kernel void compactVisiblePairs(
    device const VisibilityResult* allResults [[buffer(0)]],
    device VisibilityResult* visibleResults [[buffer(1)]],
    device atomic_uint* visibleCount [[buffer(2)]],
    constant uint& totalPairs [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    
    if (idx >= totalPairs) return;
    
    VisibilityResult result = allResults[idx];
    
    if (result.isVisible) {
        uint writeIdx = atomic_fetch_add_explicit(visibleCount, 1, 
                                                  memory_order_relaxed);
        visibleResults[writeIdx] = result;
    }
}

// Kernel: Priority scoring (Fisher Information approximation)
kernel void computeTaskingPriority(
    device const VisibilityResult* visiblePairs [[buffer(0)]],
    device const ObjectState* objects [[buffer(1)]],
    device float* priorities [[buffer(2)]],
    constant uint& numVisible [[buffer(3)]],
    constant float& currentTime [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
    
    if (idx >= numVisible) return;
    
    VisibilityResult pair = visiblePairs[idx];
    ObjectState obj = objects[pair.objectID];
    
    // Simplified Fisher Information priority
    // Priority ~ 1/(range^2) * (1 + covariance_factor)
    
    float rangeFactor = 1.0 / (pair.range_km * pair.range_km);
    float phaseFactor = (1.0 + cos(radians(pair.phase_angle_deg))) / 2.0;
    
    // Time since last observation (placeholder - would be from catalog)
    float timeSinceObs = 3600.0; // Assume 1 hour
    float custodyFactor = log(1.0 + timeSinceObs / 3600.0);
    
    // Combined priority
    priorities[idx] = rangeFactor * phaseFactor * (1.0 + custodyFactor);
}

// Bitonic sort helper (for parallel priority sorting)
kernel void bitonicSortStep(
    device VisibilityResult* data [[buffer(0)]],
    device const float* priorities [[buffer(1)]],
    constant uint& step [[buffer(2)]],
    constant uint& stage [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    
    uint partner = idx ^ (1 << stage);
    
    if (partner > idx) {
        bool shouldSwap = (priorities[idx] < priorities[partner]) == 
                         ((idx & step) == 0);
        
        if (shouldSwap) {
            // Swap both priority and result
            float tempP = priorities[idx];
            priorities[idx] = priorities[partner];
            priorities[partner] = tempP;
            
            VisibilityResult tempR = data[idx];
            data[idx] = data[partner];
            data[partner] = tempR;
        }
    }
}

// Optimized visibility for ground sensors only (simpler math)
kernel void groundSensorVisibility(
    device const ObjectState* objects [[buffer(0)]],
    device const float3* sensorPositions [[buffer(1)]], // Ground stations
    device uint8_t* visibilityMask [[buffer(2)]],       // 1 byte per pair
    constant uint& numObjects [[buffer(3)]],
    constant uint& numSensors [[buffer(4)]],
    constant float& minElevationRad [[buffer(5)]],
    uint2 gridPosition [[thread_position_in_grid]]) {
    
    uint objIdx = gridPosition.x;
    uint sensorIdx = gridPosition.y;
    
    if (objIdx >= numObjects || sensorIdx >= numSensors) return;
    
    ObjectState obj = objects[objIdx];
    float3 sensorPos = sensorPositions[sensorIdx];
    
    // Quick range check (assume 5000km limit for ground radar)
    float3 diff = obj.position - sensorPos;
    float range2 = dot(diff, diff);
    if (range2 > 25000000.0) { // 5000km squared
        visibilityMask[objIdx * numSensors + sensorIdx] = 0;
        return;
    }
    
    // Elevation check
    float3 up = normalize(sensorPos);
    float cosElev = dot(normalize(diff), up);
    
    if (cosElev < cos(minElevationRad)) {
        visibilityMask[objIdx * numSensors + sensorIdx] = 0;
        return;
    }
    
    // Earth occlusion check
    if (!lineOfSightClear(sensorPos, obj.position, 
                          EARTH_RADIUS_KM, EARTH_OBLATENESS)) {
        visibilityMask[objIdx * numSensors + sensorIdx] = 0;
        return;
    }
    
    visibilityMask[objIdx * numSensors + sensorIdx] = 1;
}
