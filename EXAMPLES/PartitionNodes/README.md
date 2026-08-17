# PartitionNodes

Regression checks for `partition` retention of floating / fixed / equalDOF nodes.

```bash
# from this directory, with OpenSeesMP on PATH or absolute path:
python3 check_partition_nodes.py /path/to/OpenSeesMP 2
python3 check_partition_nodes.py /path/to/OpenSeesMP 4
```

Cases (custom + METIS each):
- A fixed floating orphan
- B massed floating orphan
- C massless equalDOF slave to a mesh node
- D massed equalDOF slave (exactly one mass owner)
- E fixed equalDOF slave
- F floating node as equalDOF retained side
- G equalDOF between two orphans (kept together)
- H kinematic smoke: slave tracks master under Mumps
