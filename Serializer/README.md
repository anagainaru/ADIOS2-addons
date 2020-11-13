# BP4 Serializer

Implemented in `source/adios2/toolkit/format/bp/bp4/BP4Serializer.h`

The constructor takes a communicator.
Public members:
- MakeHeader - Writes a header into the metadata/data buffer
- PutProcessGroupIndex - Writes a process group index PGIndex and list of methods
- PutVariableMetadata - Put a metadata for a given variable
- PutVariablePayload - Put in buffer variable payload
- PutSpanMetadata
- SerializeMetadataInData - Serializes the metadata indices appending it into the data buffer
- CloseData - Finishes bp buffer by serializing data and adding trailing metadata
- CloseStream - Closes bp buffer for streaming mode...must reset metadata for the next step
- ResetAllIndices - Reset all metadata indices at the end of each step 
- ResetMetadataIndexTable - Reset metadata index table
- AggregateCollectiveMetadata

The way it is used by the BP4 engine:
```
format::BP4Serializer m_BP4Serializer;
m_BP4Serializer(m_Comm);

// In BeginStep
      m_BP4Serializer.m_DeferredVariables.clear();
      m_BP4Serializer.m_DeferredVariablesDataSize = 0;

// In CurrentStep
      return m_BP4Serializer.m_MetadataSet.CurrentStep;

// In PerformPuts
      if (m_BP4Serializer.m_DeferredVariables.empty())
      {
          return;
      }
      m_BP4Serializer.ResizeBuffer(m_BP4Serializer.m_DeferredVariablesDataSize,
                                   "in call to PerformPuts");
      for (const std::string &variableName : m_BP4Serializer.m_DeferredVariables)
      {
          const DataType type = m_IO.InquireVariableType(variableName);
      }

// In EndStep
      if (m_BP4Serializer.m_DeferredVariables.size() > 0)
      {
          PerformPuts();
      }
      // advances steps
      m_BP4Serializer.SerializeData(m_IO, true);
      m_BP4Serializer.ResetBuffer(m_BP4Serializer.m_Data);

      if (m_BP4Serializer.m_Parameters.CollectiveMetadata)
      {
          WriteCollectiveMetadataFile();
      }

// In WriteCollectiveMetadataFile
      m_BP4Serializer.AggregateCollectiveMetadata(
          m_Comm, m_BP4Serializer.m_Metadata, true);
      // and more functionality to write metadata into the file
          m_FileMetadataManager.WriteFiles(
              m_BP4Serializer.m_Metadata.m_Buffer.data(),
              m_BP4Serializer.m_Metadata.m_Position);
          m_FileMetadataManager.FlushFiles();

// In WriteData
dataSize = m_BP4Serializer.CloseData(m_IO);
m_FileDataManager.WriteFiles(m_BP4Serializer.m_Data.m_Buffer.data(),
                                   dataSize, transportIndex);

// In AggregateWriteData
          aggregator::MPIAggregator::ExchangeRequests dataRequests =
              m_BP4Serializer.m_Aggregator.IExchange(m_BP4Serializer.m_Data, r);

```
