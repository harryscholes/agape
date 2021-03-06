# Approximate performance of deepNF on MIPS CV taken from paper Fig. 2.
CV = {
    'level1': {
        'M_AUPR': 0.6,
        'accuracy': 0.31,
        'f1': 0.59,
        'm_AUPR': 0.73},
    'level2': {
        'M_AUPR': 0.47,
        'accuracy': 0.21,
        'f1': 0.54,
        'm_AUPR': 0.64},
    'level3': {
        'M_AUPR': 0.37,
        'accuracy': 0.21,
        'f1': 0.51,
        'm_AUPR': 0.6}}

# 10x 5-fold CV performance using code in deepNF repo. Code to generate this
# dictionary can be found below
my_CV = {
    'level1': {
        'M_AUPR': [
            0.5798497905074221, 0.5921065657550196, 0.5815431376753581,
            0.5871456339807095, 0.5846107710445478, 0.5806210756103237,
            0.5983719260295516, 0.5854614733540694, 0.5943459985340255,
            0.5951356356821152],
        'accuracy': [
            0.2938132733408324, 0.29448818897637796, 0.29403824521934757,
            0.29943757030371204, 0.29268841394825645, 0.2821147356580427,
            0.29786276715410576, 0.28166479190101235, 0.2861642294713161,
            0.3075365579302587],
        'f1': [
            0.5774994910830282, 0.5794461680808738, 0.5771242527794923,
            0.5795582176043114, 0.5732249267195746, 0.5707932609082376,
            0.5788049367448087, 0.5741171892949852, 0.5763315167082326,
            0.579830957032488],
        'm_AUPR': [
            0.7188942558844212, 0.7178060759034307, 0.7183649826616325,
            0.7214643418183545, 0.7188497410648775, 0.7065994853293351,
            0.7189226394686298, 0.7108660403006364, 0.7173948405913813,
            0.7219143523047257]},
    'level2': {
        'M_AUPR': [
            0.4734582997822961, 0.46131966591691087, 0.4602748957402941,
            0.4637101728011003, 0.44832552520847446, 0.4574353768610858,
            0.46337051747289176, 0.44723664659919554, 0.4743745498076316,
            0.4593127949983125],
        'accuracy': [
            0.20316027088036118, 0.21196388261851018, 0.21038374717832958,
            0.20316027088036118, 0.22121896162528215, 0.21128668171557563,
            0.21309255079006775, 0.21128668171557563, 0.2124153498871332,
            0.2054176072234763],
        'f1': [
            0.528013756419633, 0.5290222496699032, 0.5250744575545968,
            0.5260903148426396, 0.5226911992892245, 0.530135380664947,
            0.5235380530727177, 0.5274290125892553, 0.5216771268047955,
            0.5238896057586448],
        'm_AUPR': [
            0.6398758123066448, 0.6373748305353268, 0.6381968637119728,
            0.6352560840514924, 0.6331686373107912, 0.6406293710603486,
            0.6342872731849791, 0.6310637753498336, 0.6326698984404735,
            0.6369489457254435]},
    'level3': {
        'M_AUPR': [
            0.3838470563582621, 0.3712362521247944, 0.38159781643824137,
            0.3688475078199254, 0.3621682419057105, 0.3817827108966355,
            0.37743575298031046, 0.37507498512267917, 0.37364413698459786,
            0.3760993515555709],
        'accuracy': [
            0.23591635916359163, 0.23271832718327184, 0.238130381303813,
            0.22410824108241084, 0.2265682656826568, 0.23640836408364083,
            0.2268142681426814, 0.2265682656826568, 0.22410824108241084,
            0.22927429274292743],
        'f1': [
            0.5019335146023998, 0.499653089080115, 0.5019334662968815,
            0.49828343150210985, 0.4973392950440688, 0.5009030697819578,
            0.4976330778615255, 0.4964501671108395, 0.5016172482553977,
            0.5059125590898934],
        'm_AUPR': [
            0.6082851369994173, 0.5957901622371478, 0.6143264647171923,
            0.5979778583158104, 0.5985589745573403, 0.6079895842908615,
            0.6060163587390696, 0.6019249401485146, 0.5986936635664915,
            0.6107472008116284]}}


# import os
# import glob
# import json
# from collections import defaultdict
# import numpy as np
#
#
# def get_cv_performance(path, level):
#     path = f'{path}*level{level}'
#     files = glob(os.path.join(
#                 os.path.expandvars('$CEREVISIAEDATA'),
#                 'deepNF', 'results', f'{path}*.json'))
#     files.sort(key=lambda x: len(x.split('/')[-1].split('-')))
#     d = defaultdict(dict)
#
#     for f in files:
#         name = os.path.basename(f).split('_')
#         trial = int(name[6].replace('deepNF-SVM-trial-', ''))
#         with open(f, 'rb') as f:
#             f = json.load(f)
#             for measure, label in zip(('pr_macro', 'pr_micro', 'acc', 'fmax'),
#                                       ('M_AUPR', 'm_AUPR', 'accuracy', 'f1')):
#                 d[label][trial] = f[measure]
#     return d
#
#
# d = defaultdict(dict)
# for level in range(1, 4):
#     for measure in ('M_AUPR', 'm_AUPR', 'accuracy', 'f1'):
#         d[f'level{level}'][measure] = [
#             np.mean(i) for i in
#             get_cv_performance('180807_cd5dd8d/', level)[measure].values()]
