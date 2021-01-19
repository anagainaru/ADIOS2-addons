# SST/SSC engines

Engines for providing concurrent access to data to one or more consumers through memory or streamed over the network.

Streaing engines provide the application an API so that the data producer can mark data as generated which will make it available to be read by the consumers using the provided API in ADIOS. Each staging engine has the ability to move the data in different ways.

![Engines](docs/engines.png)

For file engines the consumer can be the same as the producer or multiple other applications running on the same network, data center or running remotely on a differnt site. The write/read operations are going through the ADIOS API which are translated underneath to POSIX calls to a storage. 

## SST

The SST  engine uses RDMA, TCP,  UDP, or shared memory to move data from a producer to consumers which can be one or multiple independent parallel applications.

Generated data is 
