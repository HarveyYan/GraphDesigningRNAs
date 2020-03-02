# Adapted from the original code base of junction tree variational autoencoder
# index_select_ND is crucial for all the indexing operations in the sparse implementation

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

