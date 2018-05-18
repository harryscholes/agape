#!/bin/bash -l
#
# Submit deepNF cross-validation jobs to SGE.
#
# This script iterates over ranges of settings for deepNF architectures and
# Gene Ontology annotations for cross-validation and submits jobs to SGE on the
# CS cluster.
#
# Usage:
#   ./cv.qsub.sh


# Job submission ranges
architectures={1..6}
ontologies={C,F,P}
levels={1..3}


# Submitter
echo STARTED

for a in $(eval echo $architectures); do
    for o in $(eval echo $ontologies); do
        for l in $(eval echo $levels); do
            echo Submitting job:
            echo architecture: $a
            echo ontology: $o
            echo level: $l
            echo
            qsub -N deepNF_cv_arch_${a}_ontology_${o}_level_${l} \
                cv.job.sh $a $o $l
done; done; done

echo DONE
