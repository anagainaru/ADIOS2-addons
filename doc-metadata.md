## Metadata aggregation scheme

The new aggregation scheme has two mandatory stages and one optional:
- The **first mandatory stage** gathers to rank 0 all the fixed size info that we normally gather with metadata, including the hashes of any metametadata and attribute blocks (but not the blocks themselves)
- At the end of the stage, we determine if we need a metametadata/attribute gather and if so who is going to contribute (the “selective” part).
- This leads to the **optional stage**, which occurs if the gather is necessary and pulls in the necessary metametadata and attribute blocks.
- Lastly, the **final mandatory stage** is gathering the metadata blocks themselves.
  Doing this in a gather by itself means that the metadata blocks end up in a contiguous memory area in rank 0 and we can write them in a single call.

*Note*
The last gather had to be changed to avoid issues on Frontier where a full-communicator gather was behaving oddly at large scales (sometimes 10-100x slower).
- In the selective metadata gather there is a limit above which we do a two-level gather of just these metadata blocks.
- A new parameter to the BP5 engine, “OneLevelGatherSizeLimit” with default value 6000 (ranks) sets this threshold.
- There’s also a Boolean parameter “UseSelectiveMetadataAggregation” that defaults to true, but if set to false we can fall back to the existing two level implementation.

## Performance tests for metadata

Small metadata tests
- 10 globals, 10 arrays, 10 attributes

Big metadata
- 1000 globals, 535 arrays, 3800 attributes

Big metadata where the attributes are written on every rank (rather than just rank 1).  
