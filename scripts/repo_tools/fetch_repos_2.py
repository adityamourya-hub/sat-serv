import urllib.request
import json
import time

repos = [
    "mit-ll/MaDDG", "Arya920/Anomaly_Detection_In_Orbit_Maneuver",
    "Tanmay-Tripathi/Maneuvers-in-orbital-data", "rrovetto/Orbital-Space-Ontology-Project",
    "ibm/spacetech-ssa", "Tomenol/deorbitation_simulation", "realsurya/Spacecraft-Reentry-Sim",
    "lcasts/REAT", "plosan/Atmospheric-Reentry", "amyinorbit/reentry.py",
    "jlev/ballistic-missile-range", "ferencdv/missile_trajectory_simulator",
    "lwcook/hypersonic-simulation", "dimitarbez/ICBMSimulator",
    "Bahrul-Rozak/Ballistic-Missile-Simulation-Visualization"
]

results = {}
for repo in repos:
    url = f"https://api.github.com/repos/{repo}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results[repo] = {
                "stars": data.get("stargazers_count"),
                "language": data.get("language"),
                "updated_at": data.get("pushed_at") or data.get("updated_at"),
                "license": data.get("license", {}).get("name") if data.get("license") else "None",
            }
    except Exception as e:
        results[repo] = {"error": str(e)}
print(json.dumps(results))
