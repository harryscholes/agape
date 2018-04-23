"""An adapted implementation of GOATOOLS `read_gaf` function including an
optional `before_date` kwarg to filter annotations before a certain date.
"""
from collections import defaultdict
import sys
from goatools.anno.gaf_reader import GafReader
from goatools.associations import get_nd, get_not

__all__ = ["my_read_gaf"]


def my_read_gaf(fin_gaf, prt=sys.stdout, before_date=None, **kws):
    """Read Gene Association File (GAF).

    # Arguments
        before_date: int, only consider annotation before this date (YYYYMMDD)

    # Returns
        dict: maps gene IDs to the GO terms it is annotated them
    """
    # keyword arguments for choosing which GO IDs to keep
    taxid2asscs = kws.get('taxid2asscs', None)
    b_geneid2gos = not kws.get('go2geneids', False)
    evs = kws.get('evidence_set', None)
    eval_nd = get_nd(kws.get('keep_ND', False))
    eval_not = get_not(kws.get('keep_NOT', False))
    # keyword arguments what is read from GAF.
    hdr_only = kws.get('hdr_only', None)  # Read all data from GAF by default
    # Read GAF file
    # Simple associations
    id2gos = defaultdict(set)
    # Optional detailed associations split by taxid and having both ID2GOs & GO2IDs
    gafobj = GafReader(fin_gaf, hdr_only, prt, **kws)
    # Optionally specify a subset of GOs based on their evidence.
    # By default, return id2gos. User can cause go2geneids to be returned by:
    #   >>> read_ncbi_gene2go(..., go2geneids=True
    for idx, ntgaf in enumerate(gafobj.associations):
        if eval_nd(ntgaf) and eval_not(ntgaf):
            if evs is None or ntgaf.Evidence_Code in evs:

                # My addition to GOATOOLS function
                if before_date:
                    return ntgaf, idx
                    if int(ntgaf.Date) > before_date:
                        continue

                taxid = ntgaf.Taxon[0]
                geneid = ntgaf.DB_ID
                go_id = ntgaf.GO_ID
                if b_geneid2gos:
                    id2gos[geneid].add(go_id)
                else:
                    id2gos[go_id].add(geneid)
                if taxid2asscs is not None:
                    taxid2asscs[taxid]['ID2GOs'][geneid].add(go_id)
                    taxid2asscs[taxid]['GO2IDs'][go_id].add(geneid)
    return id2gos  # return simple associations
