# Memory Management
It is a basic facility to manage CUDA memory dynamically  
## 1. Memory pool  
We construct a consecutive array to store our Labels and Queues.  
Specifically, we divide the one dimension array into short slices, and length of each slice is up to 100.  


