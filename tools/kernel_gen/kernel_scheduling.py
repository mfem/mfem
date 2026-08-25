class GPUConfig:
    def __init__(
        self,
        warp_size: int,
        unwrap_order=(0, 1, 2),
        nbanks: int = 32,
        op_size: int = 32,
        bank_entry_word_size: int = 4,
        allow_mcast: bool = True,
    ):
        """
        @param number of thread lanes in a single warp, 32 for NVidia, 64 for AMD, and 16 for Intel
        @param unwrap_order tuple of order which thread blocks are unwrapped (first is contiguous).
               (0,1,2) for contiguous in x, (2,1,0) for contiguous in z. Most GPUs should have (0,1,2).
        @param nbanks number of shared memory banks
        @param op_size max number of "words" in a single read op
        @param bank_entry_word_size number of bytes for each bank word
        @param allow_mcast True to allow any number of threads requesting the exact same word to be treated as a single read. Cuda compute capability 3.0 (Keplar, circa 2012) has multicast. Not sure about AMD.
        TODO: add bcast support where all threads request the same address(TODO: is this correct?)
        """
        self.warp_size = warp_size
        self.unwrap_order = unwrap_order
        self.nbanks = nbanks
        self.op_size = op_size
        self.bank_entry_word_size = bank_entry_word_size
        self.allow_mcast = allow_mcast

    def AnalyzeSMem(self, block_shape, access_func, access_size: int):
        """
        Analyze shared memory access pattern for bank conflicts
        @param block_shape thread block shape
        @param access_func returns address in shared memory being accessed (or None if skipped).
               Must be a multiple of access_size if not None.
        @param access_size Number of bytes being accessed in each entry.
               For now only supports multiples of self.bank_entry_word_size.
        @return [[access instructions] for each warp]
        each access instruction is a list of tuples (bank id, smem address)
        """
        import itertools

        tids = [
            idx[::-1]
            for idx in itertools.product(
                *[range(block_shape[i]) for i in self.unwrap_order[::-1]]
            )
        ]
        nwarps = (len(tids) + self.warp_size - 1) // self.warp_size
        warps = [
            tids[i * self.warp_size : min((i + 1) * self.warp_size, len(tids))]
            for i in range(nwarps)
        ]
        mem = [[access_func(t) for t in warp] for warp in warps]
        all_banks = [
            [
                ((a // self.bank_entry_word_size) % self.nbanks, a)
                for a in m
                if a is not None
            ]
            for m in mem
        ]
        # compute what mem ops are required
        all_accesses = []
        max_accesses = self.op_size * self.bank_entry_word_size // access_size
        for banks in all_banks:
            accesses = []
            if self.allow_mcast:
                # only need unique addresses
                addresses = set()
                ubanks = []
                for b in banks:
                    if b[1] not in addresses:
                        addresses.add(b[1])
                        ubanks.append(b)
                banks = ubanks
            # figure out the best way to break up reads to minimize bank conflicts
            slots = [dict() for i in range(self.nbanks)]
            slot_count = [0 for i in range(self.nbanks)]
            for b in banks:
                if b[1] not in slots[b[0]]:
                    slots[b[0]][b[1]] = 1
                else:
                    slots[b[0]][b[1]] += 1
                slot_count[b[0]] += 1
            nrem = len(banks)
            while nrem:
                # schedule based on which slot has the most entries remaining
                slot_order = sorted(
                    range(self.nbanks), key=lambda v: slot_count[v], reverse=True
                )
                access = []
                for i in slot_order:
                    if slot_count[i]:
                        for k, v in slots[i].items():
                            access.append((i, k))
                            slot_count[i] -= 1
                            nrem -= 1
                            if v == 1:
                                del slots[i][k]
                            else:
                                slots[i][k] -= 1
                            break
                        if len(access) == max_accesses:
                            break
                if len(access):
                    accesses.append(access)
                else:
                    raise RuntimeError("error")
            all_accesses.append(accesses)
        return all_accesses
