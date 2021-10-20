import scippneutron as scn


class LoadNexus:
    def setup(self):
        self.file_path = scn.data.get_path('PG3_4844_event.nxs')

    def time_load_event_nexus(self):
        scn.load_nexus(self.file_path)
