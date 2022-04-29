# Streaming onDemand mode

Consumers can receive steps (one step can only be consumed by one consumer) either in round robin order or on demand.

17. ``StepDistributionMode``: Default **"StepsAllToAll"**.  This value
controls how steps are distributed, particularly when there are
multiple readers.  By default, the value is **StepsAllToAll*, which
means that all timesteps are to be delivered to all readers (subject
to discard rules, etc.).  In other distribution modes, this is not the
case.  For example, in **"StepsRoundRobin"**, each step is delivered
only to a single reader, determined in a round-robin fashion based
upon the number or readers who have opened the stream at the time the
step is submitted.  In **"StepsOnDemand"** each step is delivered to a
single reader, but only upon request (with a request being initiated
by the reader doing BeginStep()).  Normal reader-side rules (like
BeginStep timeouts) and writer-side rules (like queue limit behavior) apply.
