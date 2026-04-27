package main

import (
	"fmt"
	"strconv"
)

func checksum(line string) int {
	sum := 0
	for i := 0; i < len(line)-1; i++ {
		char := line[i]
		if char >= '0' && char <= '9' {
			val, _ := strconv.Atoi(string(char))
			sum += val
		} else if char == '-' {
			sum += 1
		}
	}
	return sum % 10
}

func main() {
	l1 := "1 38779U 12051B   24108.52361111  .00000123  00000-0  00000-0 0  999"
    fmt.Printf("GSAT-10 L1: %s%d\n", l1, checksum(l1+"0"))
    l2 := "2 38779   0.0526 235.4123 0002568 123.4567 236.6543  1.00271258 4123"
    fmt.Printf("GSAT-10 L2: %s%d\n", l2, checksum(l2+"0"))
    
    i1 := "1 25544U 98067A   24108.83510521  .00016717  00000-0  30165-3 0  999"
    fmt.Printf("ISS L1: %s%d\n", i1, checksum(i1+"0"))
    i2 := "2 25544  51.6409 181.7459 0004546  51.9332 308.1750 15.4984185744955"
    fmt.Printf("ISS L2: %s%d\n", i2, checksum(i2+"0"))
}
