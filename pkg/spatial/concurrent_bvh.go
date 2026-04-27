package spatial

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"ssis/pkg/types"
)

type ConcurrentBVH struct {
	bvh     *BVH
	mu      sync.RWMutex
	version int64
}

func NewConcurrentBVH() *ConcurrentBVH {
	return &ConcurrentBVH{
		bvh:     &BVH{},
		version: 0,
	}
}

func (cbvh *ConcurrentBVH) Insert(obj BVHObject) {
	cbvh.mu.Lock()
	defer cbvh.mu.Unlock()

	cbvh.bvh.Objects = append(cbvh.bvh.Objects, obj)
	newBVH := NewBVH(cbvh.bvh.Objects)
	cbvh.bvh = newBVH
	atomic.AddInt64(&cbvh.version, 1)
}

func (cbvh *ConcurrentBVH) InsertBatch(objects []BVHObject) {
	if len(objects) == 0 {
		return
	}

	cbvh.mu.Lock()
	defer cbvh.mu.Unlock()

	cbvh.bvh.Objects = append(cbvh.bvh.Objects, objects...)
	newBVH := NewBVH(cbvh.bvh.Objects)
	cbvh.bvh = newBVH
	atomic.AddInt64(&cbvh.version, 1)
}

func (cbvh *ConcurrentBVH) QueryRange(min, max types.Vector3, tMin, tMax float64) []BVHObject {
	cbvh.mu.RLock()
	defer cbvh.mu.RUnlock()

	return cbvh.bvh.QueryRange(min, max, tMin, tMax)
}

func (cbvh *ConcurrentBVH) FindPotentialPairs() [][2]BVHObject {
	cbvh.mu.RLock()
	defer cbvh.mu.RUnlock()

	return cbvh.bvh.FindPotentialPairs()
}

func (cbvh *ConcurrentBVH) GetObjectCount() int {
	cbvh.mu.RLock()
	defer cbvh.mu.RUnlock()

	return len(cbvh.bvh.Objects)
}

func (cbvh *ConcurrentBVH) GetVersion() int64 {
	return atomic.LoadInt64(&cbvh.version)
}

type ParallelBVHBuilder struct {
	numWorkers int
}

func NewParallelBVHBuilder() *ParallelBVHBuilder {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	return &ParallelBVHBuilder{
		numWorkers: n,
	}
}

func (p *ParallelBVHBuilder) BuildParallel(objects []BVHObject) *BVH {
	if len(objects) == 0 {
		return &BVH{}
	}

	if len(objects) < 1000 {
		return NewBVH(objects)
	}

	chunks := p.splitIntoChunks(objects, p.numWorkers)
	results := make([][]mortonItem, len(chunks))

	var wg sync.WaitGroup
	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, objs []BVHObject) {
			defer wg.Done()
			results[idx] = p.computeMortonItems(objs)
		}(i, chunk)
	}
	wg.Wait()

	merged := p.mergeMortonItems(results)
	return NewBVHFromMortonItems(merged)
}

func (p *ParallelBVHBuilder) splitIntoChunks(objects []BVHObject, numChunks int) [][]BVHObject {
	chunkSize := (len(objects) + numChunks - 1) / numChunks
	chunks := make([][]BVHObject, 0, numChunks)

	for i := 0; i < len(objects); i += chunkSize {
		end := i + chunkSize
		if end > len(objects) {
			end = len(objects)
		}
		chunks = append(chunks, objects[i:end])
	}

	return chunks
}

func (p *ParallelBVHBuilder) computeMortonItems(objects []BVHObject) []mortonItem {
	if len(objects) == 0 {
		return nil
	}

	bounds := AABB{
		Min:  objects[0].GetPosition(),
		Max:  objects[0].GetPosition(),
		TMin: objects[0].GetTimestamp(),
		TMax: objects[0].GetTimestamp(),
	}

	for _, obj := range objects {
		bounds.Expand(obj.GetPosition(), obj.GetTimestamp())
	}
	bounds.ValidateBounds()

	items := make([]mortonItem, len(objects))
	for i, obj := range objects {
		items[i] = mortonItem{
			code:   mortonCode(obj.GetPosition(), obj.GetTimestamp(), bounds),
			object: obj,
		}
	}

	return items
}

func (p *ParallelBVHBuilder) mergeMortonItems(chunks [][]mortonItem) []mortonItem {
	totalLen := 0
	for _, chunk := range chunks {
		totalLen += len(chunk)
	}

	merged := make([]mortonItem, 0, totalLen)
	for _, chunk := range chunks {
		merged = append(merged, chunk...)
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].code < merged[j].code
	})

	return merged
}

func NewBVHFromMortonItems(items []mortonItem) *BVH {
	if len(items) == 0 {
		return &BVH{}
	}

	objects := make([]BVHObject, len(items))
	for i, item := range items {
		objects[i] = item.object
	}

	bounds := AABB{
		Min:  items[0].object.GetPosition(),
		Max:  items[0].object.GetPosition(),
		TMin: items[0].object.GetTimestamp(),
		TMax: items[0].object.GetTimestamp(),
	}

	for _, item := range items {
		bounds.Expand(item.object.GetPosition(), item.object.GetTimestamp())
	}
	bounds.ValidateBounds()

	root := buildBVHRecursive(items, bounds)

	return &BVH{
		Root:    root,
		Objects: objects,
		Bounds:  bounds,
	}
}

type QueryExecutor struct {
	bvh     *BVH
	queries chan QueryTask
	results chan []BVHObject
	workers int
}

type QueryTask struct {
	Min   types.Vector3
	Max   types.Vector3
	TMin  float64
	TMax  float64
	Reply chan []BVHObject
}

func NewQueryExecutor(bvh *BVH, workers int) *QueryExecutor {
	if workers < 1 {
		workers = runtime.GOMAXPROCS(0)
	}

	return &QueryExecutor{
		bvh:     bvh,
		queries: make(chan QueryTask, workers*2),
		results: make(chan []BVHObject, workers*2),
		workers: workers,
	}
}

func (qe *QueryExecutor) Start() {
	for i := 0; i < qe.workers; i++ {
		go qe.worker()
	}
}

func (qe *QueryExecutor) worker() {
	for task := range qe.queries {
		results := qe.bvh.QueryRange(task.Min, task.Max, task.TMin, task.TMax)
		task.Reply <- results
	}
}

func (qe *QueryExecutor) SubmitQuery(min, max types.Vector3, tMin, tMax float64) []BVHObject {
	reply := make(chan []BVHObject, 1)
	qe.queries <- QueryTask{
		Min:   min,
		Max:   max,
		TMin:  tMin,
		TMax:  tMax,
		Reply: reply,
	}
	return <-reply
}

func (qe *QueryExecutor) Shutdown() {
	close(qe.queries)
	close(qe.results)
}
