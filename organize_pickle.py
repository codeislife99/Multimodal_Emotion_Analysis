from mmdata import Dataloader, Dataset # we need the Dataset class for merging
mosei = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI')
mosei_facet = mosei.facet()