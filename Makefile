CC      = mpicc
CFLAGS  = -O2 -pthread

TARGETS = baseline comm_self offload

all: $(TARGETS)

baseline: baseline.c
	$(CC) $(CFLAGS) $< -o $@

comm_self: comm_self.c
	$(CC) $(CFLAGS) $< -o $@

offload: offload.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS) *.o
