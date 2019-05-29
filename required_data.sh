# Download public files

cd $AGAPEDATA

# Gene IDs
if [ ! -f schizosaccharomyces_pombe.genome.gff3 ]; then
    curl -O ftp://ftp.pombase.org/pombe/genome_sequence_and_features/gff3/schizosaccharomyces_pombe.genome.gff3.gz \
    && gunzip schizosaccharomyces_pombe.genome.gff3.gz
fi

# Gene ID, symbol, name
if [ ! -f sysID2product.tsv ]; then
    curl -O ftp://ftp.pombase.org/pombe/names_and_identifiers/sysID2product.tsv
fi

# GO
if [ ! -f gene_association.pombase ]; then
    curl -O ftp://ftp.geneontology.org/pub/go/gene-associations/gene_association.pombase.gz \
    && gunzip gene_association.pombase.gz
fi

if [ ! -f go.obo ]; then
    curl -O http://snapshot.geneontology.org/ontology/go.obo
fi

# S. pombe viability
if [ ! -f FYPOviability.tsv ]; then
    curl -O ftp://ftp.pombase.org/pombe/annotations/Phenotype_annotations/FYPOviability.tsv
fi

# BioGRID
if [ ! -f BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.164.tab2.txt ]; then
    curl -O https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-3.4.164/BIOGRID-ORGANISM-3.4.164.tab2.zip \
    && unzip BIOGRID-ORGANISM-3.4.164.tab2.zip -d Biogrid/ \
    && mv Biogrid/BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.164.tab2.txt . \
    && rm -r Biogrid/ BIOGRID-ORGANISM-3.4.164.tab2.zip
fi

# STRING
if [ ! -f 4896.protein.links.detailed.v10.5.txt ]; then
    curl -O https://stringdb-static.org/download/protein.links.detailed.v10.5/4896.protein.links.detailed.v10.5.txt.gz \
    && gunzip 4896.protein.links.detailed.v10.5.txt.gz
fi

# Gene expression meta-analysis
if [ ! -f pombeallpairs..genexp.txt ]; then
    curl -O http://bahlerweb.cs.ucl.ac.uk/meta-analysis/pombeallpairs..genexp.txt.gz \
    && gunzip pombeallpairs..genexp.txt.gz
fi

# FYPO annotations
if [ ! -f phenotype_annotations.pombase.phaf ]; then
    curl -O ftp://ftp.pombase.org/pombe/annotations/Phenotype_annotations/phenotype_annotations.pombase.phaf.gz \
    && gunzip phenotype_annotations.pombase.phaf.gz
fi
