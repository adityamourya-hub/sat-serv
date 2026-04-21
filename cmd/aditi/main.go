// cmd/aditi/main.go - ADITI 4.0 Cobra CLI Entry Point
//
// Because Cobra abstracts the command tree, the actual main.go becomes a single line.
// All command logic is in separate files (root.go, catalog.go, propagate.go, pax.go)

package main

func main() {
	Execute()
}
