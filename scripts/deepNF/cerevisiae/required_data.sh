cd $CEREVISIAEDATA

# STRING
if [ ! -f 4932.protein.links.detailed.v10.5.txt ]; then
    curl -O https://stringdb-static.org/download/protein.links.detailed.v10.5/4932.protein.links.detailed.v10.5.txt.gz \
    && gunzip 4932.protein.links.detailed.v10.5.txt.gz
fi

if [ ! -f 4932.protein.links.detailed.v9.1.txt ]; then
    curl -O http://string91.embl.de/newstring_download/protein.links.detailed.v9.1/4932.protein.links.detailed.v9.1.txt.gz \
    && gunzip 4932.protein.links.detailed.v9.1.txt.gz
fi

# GO
if [ ! -f gene_association.sgd ]; then
    curl -O http://geneontology.org/gene-associations/gene_association.sgd.gz \
    && gunzip gene_association.sgd.gz
fi

if [ ! -f go.obo ]; then
    curl -O http://snapshot.geneontology.org/ontology/go.obo
fi

# deepNF cerevisiae GO annotations for cross-validation
if [ ! -f deepNF/annotations/yeast_annotations.mat ]; then
    curl -O https://users.flatironinstitute.org/vgligorijevic/public_www/deepNF_data/yeast_annotations.mat
fi
