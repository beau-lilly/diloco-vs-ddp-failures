# CPS 390-03: Independent Semester Projects

The purpose of the independent project is to give you the opportunity to dive into something that interests you and bring something interesting back to the class. You might also take it as an opportunity to help build out infrastructure for future offerings of this course: make up a new lab, frame up missing material, or dive deeper. Become an expert in something, produce an output you can talk to others about. Hone your skills for critical thinking and presentation.

Here are some flavors of projects to consider:

- **Make the next lab:** Develop a programming exercise that illustrates a topic for a future offering of this course.
- **Deploy and measure:** Deploy an open-source service in the cloud, drive it with a workload, measure it, monitor it. 
- **Explore a transition in progress:** Investigate the current state of a technical transition (examples: gRPC over QUIC/HTTP3, DANE).
- **Frame a problem:** Frame a technical problem of interest, propose a solution, and evaluate its practicality.

Part of the purpose of this exercise is to see what you come up with.  Some of the ideas below are "from the hip" research-flavored or involve building something from scratch or are vague starter pointers to topic areas that might be interesting.    There is also a lot of value in a simpler kind of project: deploy something real, poke it, see what happens, measure what you find, and write up what you learned. GKE (Google Kubernetes Engine) is a good platform for this — you can deploy a multi-service application, kill pods, watch the scheduler reschedule them, experiment with health checks and rolling updates, and see the gap between "I understand distributed systems in theory" and "I just watched a node die and my service stayed up."    Pair them with monitoring or failure injection tools.   There are a few pointers below but
they are not limiting.

## Standard Default Project

Complete some work from the MIT distributed systems projects: see [6.824](https://pdos.csail.mit.edu/6.824/).   

---

## Cloud Experiments

Create an example use case or workload and deploy it in a commercial cloud. Use an existing workload generator to measure/evaluate it. You could experiment with auto-scaling, or use monitoring software to instrument it and analyze/illustrate its operation. As a student you have free access to cloud resources through [Google Cloud for Education](https://cloud.google.com/edu/students) credits and [Google Colab](https://colab.research.google.com/); GCP's free tier also works but requires a credit card.   Other providers offer similar arrangements.

**Message queues and event streaming:**
Kafka, NATS, and RabbitMQ are core infrastructure for asynchronous communication between services, and they make the contrast with synchronous RPC concrete and visceral. Stand up a small cluster (NATS is especially lightweight and is written in Go), push traffic through it, kill a broker, and watch what happens to ordering and delivery guarantees. These systems surface questions about exactly-once delivery, consumer groups, partitioning, and backpressure that are hard to appreciate from a textbook. 

**Other open-source systems to consider:**
Flask, Express, MongoDB, ES6, [Redis/Lettuce](https://redis.github.io/lettuce/) (see also Redis, Memorystore, and Valkey), etcd, zookeeper.

**Workload generators:**
We often evaluate systems with synthetic workload generators that emulate workloads observed in the wild. Measurement studies provide useful insights into what happens in the wild, which are often followed by workload generators to drive traffic to evaluate new systems.   Here are some examples that come to mind:

- [memtier_benchmark](https://redis.io/blog/memtier_benchmark-a-high-throughput-benchmarking-tool-for-redis-memcached/) — a high-throughput benchmarking tool for Redis/Memcached
- [DeathStarBench](https://github.com/delimitrou/DeathStarBench) — a benchmark suite of representative microservice applications, good for deploying on GKE and driving with realistic traffic

- [A Cloud-Scale Characterization of Remote Procedure Calls](https://dl.acm.org/doi/10.1145/3600006.3613156) (Google's environment) — Focus on their data analytics and internal monitoring systems. How much of this kind of analysis and monitoring can be captured with open source?  Can you build a synthetic generator that matches profiles of this kind of traffic (e.g., for simulators)? What are the opportunities for application-specific optimizations?


---

## Monitoring and Observability

These projects are natural value-adds for any cloud deployment project above. If you deploy a service in GKE, consider pairing it with one of these to deepen the investigation.

**Monitoring tools:**
For monitoring, there are a variety of tools based on [Cilium](https://github.com/cilium/cilium) and [eBPF](https://ebpf.io). These let you observe network traffic, system calls, and application behavior at the kernel level without modifying application code — a very different approach from traditional logging.

**Distributed tracing:**
Instrument a multi-service application with [OpenTelemetry](https://opentelemetry.io/) and visualize traces in [Jaeger](https://www.jaegertracing.io/) or [Zipkin](https://zipkin.io/). The goal is not just to set it up but to actually use the traces to diagnose something: find a latency bottleneck, identify a slow dependency, or understand how a request fans out across services. Deploy one of the workload generators above, generate traffic, and then play detective with the traces. This is what SREs actually do.   See also [Distributed Latency Profiling through Critical Path Tracing](https://spawn-queue.acm.org/doi/pdf/10.1145/3526967)

---

## Failure Injection

Also a natural companion to any cloud deployment project. The idea is to break things on purpose, systematically, and observe how the system degrades — or doesn't.

[Chaos Mesh](https://chaos-mesh.org/) and [LitmusChaos](https://litmuschaos.io/) are open-source chaos engineering platforms that run natively on Kubernetes. You can inject network partitions, latency spikes, pod kills, and resource contention, all from a dashboard. Deploy a service on GKE, establish a baseline with normal traffic, then inject failures and measure the impact. How does latency change? Do requests fail? Does the system recover, and how fast? Students enjoy breaking things, and the pedagogical payoff is real: you learn more about a system's reliability from watching it fail than from watching it succeed.

---

## ML Cloud Workflows

We will talk about TensorFlow, which provides an interface to accelerators such as TPUs.   It is obviously a hot topic and there is lots of infrastructure to lead you into it.   This direction is a tangent from the core course material, but it is allowable if it interests you.

The distributed systems content here is in how these systems partition and parallelize work: data parallelism, model parallelism, pipeline parallelism, and the communication patterns (all-reduce, parameter servers) that stitch them together. These are real distributed systems problems — synchronization, fault tolerance, stragglers.

Check out this book on scaling LLMs on TPUs: [Scaling Book (JAX-ML)](https://jax-ml.github.io/scaling-book/). It includes exercises and runnable code designed for [Google Colab](https://colab.research.google.com/), which provides free access to a TPU v2-8 — enough to train a small model from a document corpus and observe the parallelism and communication patterns firsthand. Students with [Google Cloud for Education](https://cloud.google.com/edu/students) credits can go further. A concrete project might be to walk through how one of the parallelism strategies works, reproduce it at small scale on Colab, and measure the communication overhead versus the compute. Or: read one of the systems papers behind distributed training (e.g., Megatron-LM, Pathways, or the original parameter server paper) and present what the distributed systems challenges are and how they're addressed.

---

## gRPC

gRPC is the standard RPC framework for microservices and it runs fine on a laptop — you can spin up multiple server processes and a client locally. The interesting feature for this course is [interceptors](https://grpc.io/docs/guides/interceptors/), which are middleware hooks on both the client and server side that can inspect and modify RPCs in flight. They're the mechanism for cross-cutting concerns: logging, authentication, retries, and — most relevant here — **failover**.

**A laptop-scale project:** Run a small set of gRPC server replicas (say, three) where one is the "leader" and the others are backups. Write a client-side interceptor that detects when the leader is unreachable (RPC returns `Unavailable`) and rebinds the connection to a backup. On the server side, a follower that receives a request meant for the leader can respond with a redirect status and the leader's address. The client interceptor handles the rebind transparently — the application code just sees its RPCs succeed. This is the pattern real Raft-based systems like etcd use, but stripped down to the failover mechanism itself.

For a more ambitious version, combine this with an actual Raft implementation (etcd/raft or hashicorp/raft) so the servers elect a leader and the client tracks it. Add a server-side interceptor that deduplicates retried requests using a (clientID, requestID) cache — this is the reply-cache/WAL component needed for exactly-once semantics across failover.

**Two approaches to routing in production:**

- **Proxy / sidecar:** Use Envoy/xDS when you want centralized L7 routing policies, easier observability, and to avoid pushing logic to all clients. This is the production-recommended approach.
- **Server-side redirect:** For Raft-style leader-aware clusters, followers reply with the leader address and clients rebind. Simpler, no proxy dependency, good for this project.

---

## Go Projects

**Channel sets and dynamic selection:**
Why did Go reject channel sets? The `select` statement has a fixed set of discrete cases. What about goroutines that handle dynamic sets of channels? There is no support for unified notifications from many channels that change dynamically, comparable to the Unix `select` system call, its successor `kqueue`, or the `MessageQueue` primitive in Android.

**Deadlines, cancellation, and tail latency:**
Flesh out how to apply Go's support for deadlines and cancellation and how it might be used to implement design patterns in [The Tail At Scale](https://research.google/pubs/the-tail-at-scale/).

---

## Propose a Lab

Some potential projects build out the basic flow among goroutines in our labs to model other interesting structures, with design patterns that avoid deadlock.
The AMO lab that we skipped is one example.   This one is like Centi but with a single front-end and a single shard.   The trick is to serve the shard as a replica group and provide leader election and the support to rebind the client to the new leader.   Rather than implement a full replication protocol, the replicas could run as goroutines that share the state in memory and acquire a mutex to access the state as leader after failover.   Clients must retransmit if they don't receive a reply (e.g., after fail-over or drops from faults injected by the autograder).   Leaders must share a reply cache.

---

## Electrotech: Energy Flow

So much of cloud infrastructure is about matching supply and demand — auto-scaling, load balancing, capacity planning. Energy systems have the same fundamental problem, but in a simpler context: the units of the resource are interchangeable and stateless (a watt-hour is a watt-hour), there is no read/write distinction, and there is no identity to track. That makes energy a clean setting to study supply/demand matching, storage policies, and cost optimization without the complexity of data consistency. The concepts transfer directly: variable load, peak capacity, utilization, producer/consumer dynamics, and the cost of over- or under-provisioning.

**The project:** Build a self-contained Go program that simulates a home energy network. Each element is a goroutine communicating over channels:

- **PV panels** produce variable supply based on a synthetic solar curve (time of day, weather noise).
- **Household loads** consume variable demand (baseline plus random spikes — the oven, the AC, the EV charger).
- **A battery** with finite capacity (kWh) and charge/discharge rate limits (kW) stores and releases energy.
- **The utility grid** buys and sells power at time-of-use prices — cheap overnight and during solar peak, expensive mornings and evenings.
- **A policy goroutine** decides in each time step: charge the battery from solar? Discharge to the house? Sell back to the grid? Buy cheap power overnight and bank it?

The interesting part is the policy. A greedy policy (use solar first, then battery, then grid) is easy to implement but leaves money on the table. A smarter policy accounts for the price schedule, forecasts of solar production and demand, and battery state-of-charge. Students can implement multiple policies and compare their cost over a simulated day or week. This is the same class of problem as cloud auto-scaling — when do you provision ahead of demand, and when do you react? — but with a tangible, intuitive domain.

For reference, this is essentially what the brains of a home battery app does (Tesla Powerwall, Enphase, etc.), but those systems are opaque about their decision-making. See also [evcc](https://github.com/evcc-io/evcc), an open-source EV charging manager written in Go that integrates with solar and time-of-use pricing — a real-world system in the same space, though much more complex than what's needed here.

---

## Web Server Authentication and DNS Security

**Why can't we bypass Certifying Authorities (PKI)?**
We talked about certificates and PKI to bind a domain name securely to a public key. These evolved on the assumption that DNS is entirely insecure — and it was! But since then, DNSSEC has rolled out digital signature chains grounded in the DNS root. Why not use DNS to distribute and certify public keys too? DNS-based Authentication of Named Entities (DANE) is a proposed standard (RFC 6698). It works by introducing a new DNS record type called TLSA for a named entity's public key (or hash). (TLSA does not stand for anything.) Even so, DANE has not caught on. Why?

There are a bunch of possible projects in this tangle of issues. It is a good story about tech ecosystem architecture and its evolution, and the collective action problems that occur as everyone gropes toward a better future.

**DNS trust:**
Users can bind their devices to trusted DNS resolvers over encrypted connections to hide their DNS lookup traffic from spying/lying network owners. VPNs exist in part for this purpose, and two other options are DNS-over-HTTPS (DoH) and DNS-over-TLS (DoT). How about a recursive resolver service that supports DANE and returns a public key (or hash) with the resolver result?

See also: [CleanDNS](https://www.cleandns.com/about-us/) — their role in terms of technical sensors/analytics/actuators and partner network.

---

## Fortifying PKI

PKI backs up DNS security, but it has inherent problems. Certified keypairs may be compromised, and the PKI hierarchy is a "rootless forest" — any CA can issue a cert for any domain name. That's great if you want to surveil somebody and can inject a fake CA root certificate into their root store, but it's a problem if you are a web property and you want to protect your users from rogue or compromised CAs. New services have evolved to improve PKI security:

- **OCSP** (Online Certificate Status Protocol) to check certificate status.
- **Transparency logs** maintain an accountable directory of certificate issuances. They are interesting in their own right and increasingly used for developer code signing (cf. SigStore). Google has released open-source software for transparency logs (RFC 6962, Trillian). These are a good entree into authenticated data structures and accountable services.

What are the privacy and operational tradeoffs of these services? ISRG, the nonprofit that runs Let's Encrypt CA, is open about its choices and rationale. They are backing out of managing transparency logs and OCSP — e.g., [RFC 6962 Logs EOL](https://letsencrypt.org/2025/08/14/rfc-6962-logs-eol). What are the concerns in this space today and how can they be addressed?

---

## Internet Governance

Interested in tech policy? There are lots of policy-adjacent topics here. Internet governance has historically been rooted in the United States and the organizations are subject to US law. Is it subject to "weaponization" in the sense of Farrell and Newman? Is more globalized governance possible or desirable? What steps have been taken; how have they turned out; what steps are planned; who is driving it? One step might be to take a look at the process around the [UN Global Digital Compact](https://www.un.org/digital-emerging-technologies/global-digital-compact).
