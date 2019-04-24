#!/usr/bin/python3
# the above uses specific Python version; allows script name in top
##!/usr/bin/env python3
# the above sources Python from $PATH

"""
usage: intronIC.py [-h] [-g GENOME] [-a ANNOTATION] -n SPECIES_NAME
                   [-q SEQUENCE_INPUT] [-e] [-s] [-nc] [-i] [-v]
                   [-m {matrix file}] [-r12 {reference U12 intron sequences}]
                   [-r2 {reference U2 intron sequences}] [--no_plot]
                   [--format_info] [-d] [-u] [-na] [-t 0-100] [-ns]
                   [--five_score_coords start stop]
                   [--three_score_coords start stop] [-bpc start stop]
                   [-r {five,bp,three} [{five,bp,three} ...]] [-b]
                   [--recursive] [--subsample_n SUBSAMPLE_N]
                   [--parallel_cv PARALLEL_CV]

intronIC (intron Interrogator and Classifier) is a script which collects all
of the annotated introns found in a genome/annotation file pair, and produces
a variety of output files (*.iic) which describe the annotated introns and
(optionally) their similarity to known U12 sequences. Without the '-m' flag,
there MUST exist a matrix file in the 'intronIC_data' subdirectory in the same
parent directory as intronIC.py, with filename 'scoring_matrices.fasta.iic'.
In the same data directory, there must also be a pair of sequence files (see
--format_info) with reference intron sequences named '[u2,
u12]_reference_set.introns.iic'

optional arguments:
  -h, --help            show this help message and exit
  -e, --use_exons       Use exon rather than CDS features to define introns
                        (default: False)
  -s, --sequences_only  Bypass the scoring system and simply report the intron
                        sequences present in the annotations (default: False)
  -nc, --allow_noncanonical
                        Do not omit introns with non-canonical splicing
                        boundaries from scoring (default: False)
  -i, --allow_multiple_isoforms
                        Include non-duplicate introns from isoforms other than
                        the longest in the scored intron set (default: False)
  -v, --allow_intron_overlap
                        Allow introns with boundaries that overlap other
                        introns from higher-priority transcripts (longer
                        coding length, etc.) to be included. This will
                        include, for instance, introns with alternative 5′/3′
                        boundaries (default: False)
  -m {matrix file}, --custom_matrices {matrix file}
                        One or more matrices to use in place of the defaults.
                        Must follow the formatting described by the
                        --format_info option (default: None)
  -r12 {reference U12 intron sequences}, --reference_u12s {reference U12 intron sequences}
                        introns.iic file with custom reference introns to be
                        used for setting U12 scoring expectation, including
                        flanking regions (default: None)
  -r2 {reference U2 intron sequences}, --reference_u2s {reference U2 intron sequences}
                        introns.iic file with custom reference introns to be
                        used for setting U12 scoring expectation, including
                        flanking regions (default: None)
  --no_plot             Do not output illustrations of intron
                        scores/distributions(plotting requires matplotlib)
                        (default: False)
  --format_info         Print information about the system files required by
                        this script (default: False)
  -d, --include_duplicates
                        Include introns with duplicate coordinates in the
                        intron seqs file (default: False)
  -u, --uninformative_naming
                        Use a simple naming scheme for introns instead of the
                        verbose, metadata-laden default format (default:
                        False)
  -na, --no_abbreviation
                        Use the provided species name in full within the
                        output files (default: False)
  -t 0-100, --threshold 0-100
                        Threshold value of the SVM-calculated probability of
                        being a U12 to determine output statistics (default:
                        90)
  -ns, --no_sequence_output
                        Do not create a file with the full intron sequences of
                        all annotated introns (default: False)
  --five_score_coords start stop
                        Coordinates describing the 5' sequence to be scored,
                        relative to the 5' splice site (e.g. position 0 is the
                        first base of the intron); half-closed interval
                        [start, stop) (default: (-3, 9))
  --three_score_coords start stop
                        Coordinates describing the 3' sequence to be scored,
                        relative to the 3' splice site (e.g. position -1 is
                        the last base of the intron); half-closed interval
                        [start, stop) (default: (-5, 4))
  -bpc start stop, --branch_point_coords start stop
                        Coordinates describing the region to search for branch
                        point sequences, relative to the 3' splice site (e.g.
                        position -1 is the last base of the intron); half-
                        closed interval [start, stop) (default: (-45, -5))
  -r {five,bp,three} [{five,bp,three} ...], --scoring_regions {five,bp,three} [{five,bp,three} ...]
                        Intron sequence regions to include in intron score
                        calculations (default: ('five', 'bp'))
  -b, --abbreviate_filenames
                        Use abbreviated species name when creating output
                        filenames (default: False)
  --recursive           Generate new scoring matrices and training data using
                        confident U12s from the first scoring pass. This
                        option may produce better results in species distantly
                        related to the species upon which the training
                        data/matrices are based, though beware accidental
                        training on false positives. Recommended only in cases
                        where clear separation between types is seen on the
                        first pass (default: False)
  --subsample_n SUBSAMPLE_N
                        Number of sub-samples to use to generate SVM
                        classifiers; 0 uses the entire training set and should
                        provide the best results; otherwise, higher values
                        will better approximate the entire set at the expense
                        of speed (default: 0)
  --parallel_cv PARALLEL_CV
                        Number of parallel processes to use during cross-
                        validation; increasing this value will reduce runtime
                        but may result in instability due to outstanding
                        issues in scikit-learn (default: 1)

required arguments (-g, -a | -q):
  -g GENOME, --genome GENOME
                        Genome file in FASTA format (gzip compatible)
                        (default: None)
  -a ANNOTATION, --annotation ANNOTATION
                        Annotation file in gff/gff3/gtf format (gzip
                        compatible) (default: None)
  -n SPECIES_NAME, --species_name SPECIES_NAME
                        Binomial species name, used in output file and intron
                        label formatting. It is recommended to include at
                        least the first letter of the species, and the full
                        genus name since intronIC (by default) abbreviates the
                        provided name in its output (e.g. Homo_sapiens -->
                        HomSap) (default: None)
  -q SEQUENCE_INPUT, --sequence_input SEQUENCE_INPUT
                        Provide intron sequences directly, rather than using a
                        genome/annotation combination. Must follow the
                        introns.iic format (see README for description)
                        (default: None)
"""

import argparse
import copy
import logging
import math
import os
import re
import random
import sys
import time
import gzip
import numpy as np
import warnings

# hacky way to ignore annoying sklearn warnings
# (https://stackoverflow.com/a/33616192/3076552)
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from scipy import stats as pystats
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from functools import partial
from itertools import islice
from operator import attrgetter
from hashlib import md5
# from sklearn.metrics import roc_curve, roc_auc_score
# from sklearn.cluster import SpectralClustering
from sklearn import svm, preprocessing
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score

# improve sklearn parallel performance/stability
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

# check for the plotting library required to produce
# optionall figures
try:
    import matplotlib
    matplotlib.use('Agg') # allow to run without X display server
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    CAN_PLOT = True
except ModuleNotFoundError:
    CAN_PLOT = False

warnings.filterwarnings(action='ignore')

# Classes ####################################################################

class GFFLineInfo(object):
    """
    Takes a gff3/gtf/gff annotation line, and returns available metadata
    about it.

    """
    def __init__(self, line, line_number):
        self.bits = self.__split_on_tabs(line)
        try:
            self.region = self.bits[0]
            try:
                self.start = min(map(int, self.bits[3:5]))
                self.stop = max(map(int, self.bits[3:5]))
            except ValueError:
                self.start = None
                self.stop = None
            self.strand = self.bits[6]
            if self.strand not in ('+', '-'):
                self.strand = '+'
            if self.bits[7] in ['0', '1', '2']:
                self.phase = int(self.bits[7])
            else:
                self.phase = '.'
            self.infostring = self.bits[8]
            self.feat_type = self.bits[2].lower()
            self.parent = self.get_parent()
            self.name = self.get_ID()
            self.line_number = line_number
        except TypeError:
            raise

    @staticmethod
    def __split_on_tabs(l, n=9):
        """
        Checks for a valid line (does not start with #).

        Splits valid line on tabs, and returns a list of the bits
        if the list length is <<n>>. Setting <<n>> to None will return
        the split line regardless of length.

        """
        if l.startswith('#'):
            return None
        l = l.strip()
        columns = l.split("\t")
        if n and len(columns) < n:
            return None
        return columns

    @staticmethod
    def __field_match(infostring, tags, delimiter, tag_order=False):
        if tag_order:
            # check for first match of tags in order
            try:
                tags = [next(t for t in tags if t.lower() in infostring.lower())]
            except StopIteration:
                return None
        info_bits = infostring.split(delimiter)
        try:
            match = next(
                e for e in info_bits
                if any(p.lower() in e.lower() for p in tags))
        except StopIteration:  # no matches found
            return None
        if "=" in match:
            substring = match.split("=")[1]
        else:
            substring = match.split()[1]
        return substring.strip("\"")


    def get_type(self, delimiter=';'):
        """
        Classifies annotation lines into type categories,
        taking into account edge cases like 'processed transcript'
        and 'pseudogenic transcript'.

        """
        og_type = self.bits[2].lower()
        if og_type == 'mrna':
            og_type = 'transcript'
        if og_type in ('gene', 'transcript', 'exon', 'cds'):
            return og_type

        disqualifying = ['utr', 'start', 'stop']
        if any(kw in og_type for kw in disqualifying):
            return og_type

        # Not an obvious type, so search for features of transcripts
        # and genes in infostring to try to infer type

        # check for explicit mention of transcript in ID
        try:
            id_string = next(
                (f for f in delimiter.split(self.infostring)
                if f.startswith("ID")))

            if any(tag in id_string for tag in ('transcript', 'mrna')):
                return 'transcript'
        except StopIteration:
            pass

        gene_tags = ["gene_id", "geneId"]
        transcript_tags = ["transcriptId", "transcript_ID"]
        # Transcripts first because genes shouldn't have transcript IDs,
        # but transcripts may have gene IDs
        for ftype, tags in zip(
                ['transcript', 'gene'], [transcript_tags, gene_tags]
        ):
            match = self.__field_match(self.infostring, tags, delimiter)
            if match:
                return ftype
            else:
                return og_type


    def get_ID(self, delimiter=";"):
        """
        Finds the ID of a given annotation file line.

        """
        # first, do it the easy way
        prefix = None
        infostring = self.infostring
        match = self.__field_match(infostring, ["ID="], delimiter)
        if match:
            return match

        # Constrain feature types to simplify indexing
        feat_type = self.feat_type
        if feat_type == "mrna":
            feat_type = "transcript"
        # all get lowercased in the comparison
        # if is no 'ID=', should reference self via others
        gene_tags = ["ID=", "gene_id", "geneId"]
        transcript_tags = ["ID=", "transcriptId", "transcript_ID"]
        tag_selector = {
            "gene": gene_tags,
            "transcript": transcript_tags
        }
        try:
            tags = tag_selector[feat_type]
        except KeyError:
            # get any ID available, prepended with the feature type
            # to keep different features of the same transcript unique
            prefix = self.feat_type
            tags = ['transcriptID', 'transcript_ID', 'gene_ID', 'geneID']

        match = self.__field_match(
            infostring, tags, delimiter, tag_order=True)

        # if nothing matches, return infostring if there's only
        # one tag in it (common for gtf parent features)
        if match is None and infostring.count(";") < 2:
            match = infostring.split(";")[0]

        if prefix:
            match = '{}_{}'.format(prefix, match)

        return match


    def get_parent(self, delimiter=";"):
        """
        Retrieves parent information from an annotation line.

        """
        feat_type_converter = {"cds": "exon", "mrna": "transcript"}
        feat_type = self.feat_type
        if feat_type in feat_type_converter:
            feat_type = feat_type_converter[feat_type]
        child_tags = [
            "Parent=", "transcript_ID",
            "transcriptId", "proteinId", "protein_ID"
        ]
        transcript_tags = ["Parent=", "gene_ID", "geneId"]
        gene_tags = ["Parent="]
        tag_selector = {
            "gene": gene_tags,
            "transcript": transcript_tags,
            "exon": child_tags
        }
        try:
            tags = tag_selector[feat_type]
        except KeyError:
            tags = list(set(child_tags + transcript_tags))
        infostring = self.infostring
        match = self.__field_match(infostring, tags, delimiter, tag_order=True)
        if not match and feat_type == "transcript":
            match = self.get_ID()

        return match


class GenomeFeature(object):
    """
    Features that all genomic entities should
    have.

    >start< and >stop< are always relative to positive strand, i.e.
    >start< is always less than >stop<.

    """
    count = 1

    # __slots__ prevents objects from adding new attributes, but it
    # significantly reduces the memory footprint of the objects
    # in use. Idea from http://tech.oyster.com/save-ram-with-python-slots/

    __slots__ = [
        'line_number', 'region', 'start', 'stop', 'parent_type',
        'strand', 'name', 'parent', 'seq', 'flank', 'feat_type', 'phase',
        'upstream_flank', 'downstream_flank', 'family_size', 'unique_num'
    ]

    def __init__(
            self, line_number=None, region=None,
            start=None, stop=None, parent_type=None,
            strand=None, name=None, parent=None,
            seq=None, flank=0, feat_type=None,
            upstream_flank=None, downstream_flank=None,
            family_size=None, phase=None
    ):
        self.region = region
        self.start = start
        self.stop = stop
        self.strand = strand
        self.name = name
        self.unique_num = self.__class__.count
        self.feat_type = feat_type
        self.parent = parent
        self.parent_type = parent_type
        self.family_size = family_size
        self.line_number = line_number
        self.phase = phase
        self.seq = seq
        self.flank = flank
        self.upstream_flank = upstream_flank
        self.downstream_flank = downstream_flank

    @property
    def length(self):
        """
        Returns the length of the object, preferentially
        inferring it from the start and stop coordinates,
        then from the length of the full sequence.

        """
        if not (self.start and self.stop):
            if not self.seq:
                return None
            else:
                return len(self.seq)
        return abs(self.start - self.stop) + 1

    def get_coding_length(self, child_type="cds"):
        """
        Returns an integer value of the aggregate
        length of all children of type child_type.

        If child_type is None, returns aggregate
        length of all children

        """
        total = 0
        while True:
            # while children themselves have children, recurse
            try:
                children = [c for c in self.children if c.children]
                for child in children:
                    total += child.get_coding_length(child_type)
            except AttributeError:
                try:
                    children = self.get_children(child_type)
                    total += sum(c.length for c in children)
                except AttributeError:  # if was called on exon or cds objects
                    total += self.length
                break
            break
        return total

    def set_family_size(self):
        """
        Assigns family_size attribute to all children.

        """
        try:
            all_children = self.children
        except AttributeError:
            return
        child_types = set([c.feat_type for c in all_children])
        for ct in child_types:
            siblings = [c for c in all_children if c.feat_type == ct]
            sibling_count = len(siblings)
            for sib in siblings:
                sib.family_size = sibling_count
                sib.set_family_size()

    def compute_name(self, parent_obj=None, set_attribute=True):
        """
        Returns a generated unique ID for the object
        in question. Passing a parent object should
        be preferred, as the ID is otherwise not
        particularly informative (but is unique!)

        """
        parent = self.parent
        parent_type = self.parent_type
        # Make a coord_id relative to parent's contents if possible
        try:
            # 1-based indexing
            unique_num = parent_obj.children.index(self) + 1
        # Otherwise, make relative to all made objects of same type
        except AttributeError:
            # every type should have this
            unique_num = "u{}".format(self.unique_num)
        coord_id = ("{}:{}_{}:{}".format(parent_type,
                                    parent,
                                    self.feat_type,
                                    unique_num))
        if set_attribute:
            setattr(self, "name", coord_id)
        return coord_id

    def update(self, other, attrs=None):
        """
        Updates attributes based on another object and
        optional attribute filter list

        """
        attrs_to_use = vars(other)  # use all if not attrs
        if attrs:
            attrs_to_use = {a: v for a, v in attrs_to_use.items() if a in attrs}
        for key, value in attrs_to_use.items():
            if hasattr(self, key):  # don't make new attributes
                setattr(self, key, value)


    def get_seq(self, region_seq=None, start=None, stop=None,
                flank=0, strand_correct=True):
        """
        Retrieves object's sequence from its parent
        sequence, with optional flanking sequence.

        If >strand_correct<, will return strand-
        corrected (e.g. reverse-complemented) sequence.

        """
        if region_seq is None:
            return self.seq

        # Assign class defaults if not values
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop

        # Correct for 1-based indexing in start and stop
        start -= 1
        # avoid negative indices
        start = max(start, 0)
        # Pull sequence and reverse if necessary
        seq = region_seq[start:stop]
        if strand_correct and self.strand == '-':
            seq = reverse_complement(seq)
        if flank > 0:
            up_flank = self.upstream_seq(region_seq, flank)
            down_flank = self.downstream_seq(region_seq, flank)
            seq = [up_flank, seq, down_flank]

        return seq

    def upstream_seq(self, region_seq, n, strand_correct=True):
        """
        Get sequence of n length from upstream of
        feature start, relative to coding direction.

        """
        if self.strand == "-":
            start = self.stop
            stop = start + n
        else:
            stop = self.start - 1
            start = max(stop - n, 0) # no negative indices
        seq = region_seq[start:stop]
        if strand_correct and self.strand == '-':
            seq = reverse_complement(seq)

        return seq

    def downstream_seq(self, region_seq, n, strand_correct=True):
        """
        Get sequence of n length from downstream of
        feature start, relative to coding direction.

        """
        if self.strand == "-":
            stop = self.start - 1
            start = max(stop - n, 0) # no negative indices
        else:
            start = self.stop
            stop = start + n
        seq = region_seq[start:stop]
        if strand_correct and self.strand == '-':
            seq = reverse_complement(seq)

        return seq


class Parent(GenomeFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.children = []
        self._intronator = intronator

    def get_children(self, child_type=None):
        """
        Returns a list of all children of type >child_type<.

        If >child_type< not specified, returns all children.

        """
        if not child_type:
            selected = self.children
        else:
            selected = [c for c in self.children if c.feat_type == child_type]
        return selected

    def get_introns(self, child_type):
        """
        Returns all introns based on >child_type<,
        including any duplicates across children.

        """
        introns = []
        filtered_children = []
        intron_count = 1
        try:
            filtered_children = [child for child in self.children if
                                 child.feat_type == child_type]
        except AttributeError:
            return introns
        if not filtered_children:
            try:
                for child in self.children:
                    introns += child.get_introns(child_type)
                return introns
            except AttributeError:
                return introns
        coding_length = self.get_coding_length(child_type)
        for indx, intron in enumerate(
            self._intronator(filtered_children), start=1):
            intron.parent_length = coding_length
            intron.index = indx
            introns.append(intron)

        return introns


class Gene(Parent):
    def __init__(self, parent=None, **kwargs):
        super().__init__(**kwargs)
        self.__class__.count += 1
        self.feat_type = "gene"
        self.parent = parent
        self.parent_type = None


class Transcript(Parent):
    def __init__(self, parent=None, **kwargs):
        super().__init__(**kwargs)
        self.__class__.count += 1
        self.feat_type = "transcript"
        self.parent_type = "gene"
        if not parent:
            self.parent = self.name
        else:
            self.parent = parent


class Exon(GenomeFeature):
    def __init__(self, feat_type="exon", parent=None,
                 grandparent=None, **kwargs):
        super().__init__(**kwargs)
        self.__class__.count += 1
        self.parent_type = "transcript"
        self.feat_type = feat_type
        self.parent = parent
        self.grandparent = grandparent


class Intron(GenomeFeature):
    __slots__ = [
        '__dict__', 'bp_raw_score', 'bp_region_seq',
        'bp_seq', 'bp_start', 'bp_stop', 'bp_z_score', 'corrected',
        'downstream_exon', 'downstream_flank', 'duplicate',
        'dynamic_tag', 'family_size', 'feat_type',
        'five_display_seq', 'five_raw_score',
        'five_score_coords', 'five_seq', 'five_start', 'five_stop',
        'five_z_score', 'flank', 'fractional_position',
        'grandparent', 'index', 'line_number', 'longest_isoform',
        'name', 'noncanonical', 'omitted', 'overlap', 'parent',
        'parent_type', 'phase', 'protein_coding', 'region',
        'seq', 'start', 'stop', 'strand', 'type_id',
        'three_display_seq', 'relative_score',
        'three_score_coords', 'three_seq', 'three_z_score',
        'three_raw_score', 'u12_matrix', 'u2_matrix', 'unique_num',
        'upstream_exon', 'upstream_flank', 'dnts', 'svm_score'
    ]

    def __init__(self, parent=None, grandparent=None, **kwargs):
        # Set certain attrs from parent class
        super().__init__(**kwargs)
        self.__class__.count += 1
        # Set other intron-only attrs
        # Inherit from transcript
        self.feat_type = "intron"
        self.parent_type = "transcript"
        self.parent = parent
        self.grandparent = grandparent
        self.five_raw_score = None
        self.five_z_score = None
        self.bp_raw_score = None
        self.bp_z_score = None
        self.index = None  # in coding orientation
        # self.five_seq = None
        self.five_seq = None
        self.three_seq = None
        self.bp_seq = None
        self.bp_region_seq = None
        self.five_start = None
        self.five_stop = None
        self.bp_start = None
        self.bp_stop = None
        self.omitted = False
        self.corrected = False
        self.duplicate = False
        self.overlap = None
        self.longest_isoform = None
        self.dynamic_tag = set()
        self.noncanonical = False
        self.upstream_exon = None
        self.downstream_exon = None
        self.upstream_flank = None
        self.downstream_flank = None
        # self.phase = None
        self.fractional_position = '.'
        self.five_score_coords = None
        self.three_score_coords = None
        self.five_display_seq = None
        self.three_display_seq = None
        self.u2_matrix = None
        self.u12_matrix = None
        self.three_raw_score = None
        self.three_z_score = None
        self.dnts = None
        self.svm_score = None
        self.relative_score = None
        self.type_id = None

    @classmethod
    def from_exon_pair(cls, up_ex, down_ex):
        """
        Takes a pair of Exon objects and builds an intron
        based on their information.

        """
        # Infer intron coordinates
        start = min(up_ex.stop, down_ex.stop) + 1
        stop = max(up_ex.start, down_ex.start) - 1
        # Get applicable attributes from one of the defining coding objects
        strand = up_ex.strand
        parent = up_ex.parent
        grandparent = up_ex.grandparent
        region = up_ex.region
        # derive phase from upstream exon (CDS) phase annotation
        # (if available)
        if up_ex.phase != '.':  # default phase value if not present
            phase = (up_ex.length - up_ex.phase) % 3
        else:
            phase = '.'
        fam = up_ex.family_size - 1  # intron number is exon number - 1
        # average the line numbers of the children that define each intron
        # to enable tie-breaking downstream when deciding which duplicates
        # to exclude (for those whose parents have equal length)
        line_number = sum([x.line_number for x in (up_ex, down_ex)]) / 2
        return cls(start=start, stop=stop, strand=strand, family_size=fam,
                   parent=parent, grandparent=grandparent, region=region,
                   line_number=line_number, phase=phase)


    def get_rel_coords(self, relative_to, relative_range):
        """
        Calculates and retrieves a pair of genomic sequence coordinates
        based on input relative to the five-prime or three-prime
        end of the sequence. Returns a set of adjusted coords.

        e.g. get_rel_coords("five", "five", (0, 12)),
             get_rel_coords("bp", "three", (-45, -5))
        """

        def __upstream(ref, x, strand):
            x = abs(x)
            if strand == "-":
                return ref + x
            else:
                return ref - x

        def __downstream(ref, x, strand):
            x = abs(x)
            if strand == "-":
                return ref - x
            else:
                return ref + x

        start = self.start
        stop = self.stop
        strand = self.strand
        # Begin orientation gymnastics
        if strand == "-":
            start, stop = stop, start
        if relative_to == "five":
            ref_point = start
        else:
            ref_point = stop
        new_coords = []
        for n in relative_range:  # each number can be + or -
            if n < 0:
                new_coords.append(__upstream(ref_point, n, strand))
            else:
                new_coords.append(__downstream(ref_point, n, strand))
        new_coords = tuple(sorted(new_coords))
        # setattr(self, seq_name, new_coords)
        return new_coords

    def get_name(self, special='?'):
        """
        Build a unique name (not including score) from metadata.

        """
        if self.name is not None:
            return self.name
        if self.omitted:
            omit_tag = ';[o:{}]'.format(self.omitted)
        else:
            omit_tag = ''
        if self.dynamic_tag:
            dyn_tag = ';{}'.format(';'.join(sorted(self.dynamic_tag)))
        else:
            dyn_tag = ''
        if SIMPLE_NAME is True:
            return '{}-i_{}{}{}'.format(SPCS, self.unique_num, omit_tag, dyn_tag)
        elements = [
            self.grandparent, self.parent,
            self.index, self.family_size, omit_tag]
        tags = [e if e is not None else special for e in elements]
        tags.append(dyn_tag)  # compatibility with earlier Python 3s
        name = "{}-{}@{}-intron_{}({}){}{}".format(SPCS, *tags)
        # setattr(self, "name", name)
        return name

    def get_label(self, special='?'):
        """
        Builds a unique intron label from metadata

        """
        # if self.omitted:
        #     setattr(self, 'relative_score', 0)
        if self.relative_score is not None:
            # clipping prevents rounding from pushing introns over the
            # u12 boundary
            # *clip* float to 3 places (without rounding)
            truncated = math.floor(self.relative_score * 1000) / 1000
            score = '{}%'.format(truncated)
            # *round* float to 4 places
            # rel_score = '{:.4f}%'.format(self.relative_score)
        else:
            score = None
        if not self.name:
            self.name = self.get_name()
        label = ('{};{}'
                 .format(*[e if e is not None else special for e in
                           [self.name, score]]))
        return label


    def omit_check(
        self, min_length, allow_noncanon=False,
        allow_overlap=False, longest_only=True):
        """
        Checks an intron object for omission criteria, and sets
        the >omitted< attribute accordingly.

        """
        omit_tags = {
            'short': 's',
            'ambiguous sequence': 'a',
            'noncanonical': 'n',
            'coordinate overlap': 'v',
            'not in longest isoform': 'i'
        }
        scoring_regions = ['five_seq', 'three_seq']
        omission_reason = None
        if self.length < min_length:
            omission_reason = 'short'
        elif any(valid_chars(getattr(self, region)) is False
                 for region in scoring_regions):
            omission_reason = 'ambiguous sequence'
        # check if there is sufficiently long sequence in the
        # bp region to score at least one bp motif
        elif longest_match(self.bp_region_seq) < BP_MATRIX_LENGTH:
                omission_reason = 'ambiguous sequence'
        elif not allow_noncanon and self.noncanonical:
            omission_reason = 'noncanonical'
        elif longest_only and self.longest_isoform is False:
            omission_reason = 'not in longest isoform'
        elif allow_overlap is False and self.overlap:
            omission_reason = 'coordinate overlap'
        if omission_reason:
            setattr(self, 'omitted', omit_tags[omission_reason])

# /Classes ###################################################################

# Functions ##################################################################

def check_thresh_arg(t):
    """
    Used in argument parsing to reject malformed threshold values

    """
    t = float(t)
    if not 0 <= t <= 100:
        raise argparse.ArgumentTypeError("'{}' is not within the range 0-100".
                                         format(t))
    return t


def reverse_complement(seq):
    """
    Returns reverse complement of seq, with
    any non-ACTG characters replaced with Ns

    """
    transform = {'A': 'T',
                 'T': 'A',
                 'C': 'G',
                 'G': 'C',
                 'N': 'N'}
    try:
        comp = [transform[e] for e in seq]
    except KeyError:  # non-ATCGN characters in seq
        seq = [e if e in "ACTGN" else "N" for e in seq]
        comp = [transform[e] for e in seq]
    rev_comp = comp[::-1]
    rev_comp_string = ''.join(rev_comp)
    return rev_comp_string


def flex_open(filename):
    """
    A generator of lines from a variety of
    file types, including compressed files.

    """
    magic_dict = {
        b'\x1f\x8b\x08': partial(gzip.open, mode='rt')
        # "\x42\x5a\x68": "bz2",
        # "\x50\x4b\x03\x04": "zip"
        }

    max_len = max(len(x) for x in magic_dict)

    open_func = None

    # Try opening the file in binary mode and reading the first
    # chunk to see if it matches the signature byte pattern
    # expected for each compressed file type

    with open(filename, 'rb') as f:
        file_start = f.read(max_len)
    for magic, func in magic_dict.items():
        if file_start.startswith(magic):
            open_func = func

    if open_func:
        return open_func(filename)
    else:
        return open(filename)


def fasta_parse(fasta, delimiter=">", separator="", trim_header=True):
    """
    Iterator which takes FASTA as input. Yields
    header/value pairs. Separator will be
    used to join the return value; use separator=
    None to return a list.

    If trim_header, parser will return the
    FASTA header up to the first space character.
    Otherwise, it will return the full, unaltered
    header string.

    """
    header, seq = None, []
    with flex_open(fasta) as f:
        for line in f:
            if line.startswith(delimiter):
                if header:  # associate accumulated seq with header
                    if separator is not None:
                        seq = separator.join(str(e) for e in seq)
                    yield header, seq
                # Assign a new header
                header = line.strip().lstrip(delimiter)
                if trim_header:
                    header = header.split()[0]
                # Clear seq for new round of collection
                seq = []
            elif line.startswith('#'):
                continue
            else:
                if line.strip():  # don't collect blank lines
                    seq.append(line.rstrip('\n'))
        if separator is not None:  # make string
            seq = separator.join(str(e) for e in seq)
        yield header, seq


def get_runtime(start_time, p=3):
    """
    Takes a start time and optional decimal precision p,
    returns a string of the total run-time until current
    time with appropriate units.

    """
    total = time.time() - start_time  # start with seconds
    divided = total/60.0
    if divided < 2:
        run_time = total
        units = "seconds"
    elif divided < 60:
        run_time = divided
        units = "minutes"
    else:
        run_time = divided/60.0
        units = "hours"
    rounded = round(run_time, p)
    return "{} {}".format(rounded, units)


def abbreviate(species_name, n=3, separator=''):
    """
    Make a truncated species name from a longer
    version, of the pattern Gs{n}, where G is the
    first letter of the genus, and s is at least
    the first n letters of the species, or from
    there until a consonant if strict=False

    e.g. H.sapiens --> HomSap
         Arabidopsis_thaliana --> AraTha

    If n=None, will keep full species suffix and
    only abbreviate first name, e.g.

    Gallus_gallus --> GGallus

    """
    species_name = species_name.lower()
    bits = re.split(r"\W|_", species_name)
    if len(bits) == 1:  # no special character in string
        bit = bits[0]
        bits = [bit[0], bit[1:]]
    genus = bits[0][:n]
    genus = genus[0].upper() + genus[1:]
    sp = bits[1]
    species = sp[0].upper() + sp[1:n]
    abbreviated = separator.join([genus, species])

    return abbreviated


def load_external_matrix(matrix_file):
    """
    Makes matrices using the following file format:
    >maxtrix_name
    A       C       G       T
    0.29	0.24	0.26	0.21
    0.22	0.33	0.19	0.26
    0.27	0.18	0.04	0.51
    0	    0	    1	    0
    0	    0	    0	    1
    0.99	0.004	0.001	0.005
    0.004	0.001	0.005	0.99
    0.004	0.99	0.001	0.005
    0.004	0.99	0.001	0.005
    0.009	0.02	0.001	0.97
    0.01	0.06	0.02	0.91
    0.05	0.22	0.08	0.65
    0.19	0.29	0.2	    0.32

    Returns a dictionary of the format
    {matrix_classification: {base: [frequencies]}}

    """
    def __name_parser(matrix_name):
        """
        Will attempt to interpret a matrix name within the matrix
        file header for categorization into a tuple, describing
        region and intron type (e.g. ("u12", "bp", "gtag")

        Returns a tuple of (subtype, region, boundaries)

        """
        subtypes = {
            "u12": ["u12", "12", "minor"],
            "u2": ["u2", "major"]
        }
        regions = {
            "five": ["five", "5"],
            "bp": ["bp", "branch-point"],
            "three": ["three", "3"]
        }
        boundaries = {
            "atac": ["at-ac", "atac"],
            "gtag": ["gt-ag", "gtag"],
            "gcag": ["gc-ag", "gcag"]
        }
        name_bits = []
        for cat in [subtypes, boundaries, regions]:
            bit = next(k for k, v in cat.items() if
                       any(subv in matrix_name for subv in v))
            name_bits.append(bit)

        # add an optional version tag to the name if present
        matrix_version = re.findall('v.([^_\W]+)', matrix_name)
        if matrix_version:
            name_bits.append(matrix_version[0])

        return tuple(name_bits)

    matrices = {}
    for name, rows in fasta_parse(
        matrix_file,
        separator=None,
        trim_header=False
    ):
        try:
            start_bit = next(e for e in name.split() if 'start=' in e)
            start_index = int(start_bit.split('=')[1])
        except StopIteration:
            start_index = 0
        formatted_name = __name_parser(name.split()[0])
        matrices[formatted_name] = defaultdict(dict)
        # first row is bases in order
        bases = [b for b in rows.pop(0).split() if b in 'AGCT']
        base_index = {}
        for i, r in enumerate(rows, start=start_index):
            freqs = [float(f) for f in r.split()]
            for base, freq in zip(bases, freqs):
                matrices[formatted_name][base][i] = freq

    return matrices


def add_pseudos(matrix, pseudo=0.001):
    """
    Apply a pseudo-count of ^pseudo to every frequency in a
    frequency matrix (to every number in each keys' values)

    """
    with_pseudos = {}
    for key, value in matrix.items():
        if isinstance(value, dict):
            with_pseudos[key] = add_pseudos(value, pseudo)
        else:
            # with_pseudos[key] = [(float(f) + pseudo) for f in value]
            with_pseudos[key] = float(value) + pseudo
    return with_pseudos


def average_matrices(a, b):
    outerdict = {}
    for key, a_bases in a.items():
        if key not in b:
            outerdict[key] = a_bases
            continue
        matrix = defaultdict(dict)
        for base, a_freqs in a_bases.items():
            b_freqs = b[key][base]
            for position, a_freq in sorted(a_freqs.items()):
                try:
                    b_freq = b_freqs[position]
                    avg_freq = (a_freq + b_freq) / 2
                    matrix[base][position] = avg_freq
                except KeyError:
                    continue
        outerdict[key] = matrix

    return outerdict


def format_matrix(matrix, label="frequencies"):
    """
    Formats the contents of a matrix in FASTA
    format, with the first line being the
    order of characters, {index_order}, and the
    following lines containing the frequency of
    each character; each position in the sequence
    has its own line in the matrix.

    {label} is used as the header for the frequencies
    entry.

    example output:

    >{index_order}
    A C G T
    >{label}
    0.25 0.5 0.25 0.0
    0.0 0.75 0.25 0.0
    ...

    """
    string_list = []
    characters = sorted(matrix.keys())
    freq_index = defaultdict(list)
    for character, frequencies in sorted(matrix.items()):
        for i, e in sorted(frequencies.items()):
            freq_index[i].append(str(e))
    character_order = '\t'.join(characters)
    # string_list.append(">index_order\n{}".format(character_order))
    string_list.append(">{}\n{}".format(label, character_order))
    for i, freqs in sorted(freq_index.items()):
        string_list.append('\t'.join(freqs))
    return '\n'.join(string_list)


def introns_from_flatfile(
    flatfile,
    five_score_coords,
    three_score_coords,
    bp_coords,
    allow_noncanon,
    allow_overlap,
    hashgen=False,
    type_id=None):
    """
    Build a list of Intron objecfts from a reference file in
    introns.iic format

    """
    ref_introns = []
    auto_name = 'auto_intron_'
    auto_int = 0
    with flex_open(flatfile) as flat:
        for line in flat:
            bits = line.strip().split('\t')
            name, _, _, five, int_seq, three = bits[:6]
            if not name or name == '-':
                name = auto_name + str(auto_int)
                auto_int += 1
            # make intron coordinates relative to the combined seq
            intronic_start = len(five) + 1
            intronic_stop = len(int_seq) + len(five)
            flank_size = max(len(five), len(three))
            seq = five + int_seq + three
            new_intron = Intron(
                name=name,
                start=intronic_start,
                stop=intronic_stop
            )
            # Set sub-sequence attributes for each intron object
            new_intron = assign_seqs(
                new_intron,
                seq,
                flank_size,
                five_score_coords,
                three_score_coords,
                # five_score_length,
                bp_coords
            )
            new_intron.omit_check(
                MIN_INTRON_LENGTH,
                allow_noncanon=allow_noncanon,
                allow_overlap=allow_overlap
            )
            if hashgen:
                new_intron.md5 = md5(
                    new_intron.seq.encode('utf-8')).digest()
            new_intron.seq = None

            if type_id:
                new_intron.type_id = type_id

            yield new_intron


def get_reference_introns(ref_file, five_score_coords, three_score_coords,
                       bp_coords, type_id=None):
    """
    Build a list of Intron objects from a reference file in
    introns.iic format

    """
    refs = introns_from_flatfile(
        ref_file,
        five_score_coords,
        three_score_coords,
        bp_coords,
        allow_noncanon=True,
        allow_overlap=True,
        type_id=type_id)
    
    refs = list(refs)

    ref_introns = [i for i in refs if not i.omitted]

    omitted_refs = len(refs) - len(ref_introns)

    if omitted_refs:
        write_log(
            '{} reference introns omitted from {}',
            omitted_refs, ref_file)

    return ref_introns


def make_feat_instance(line_info, feat_type=None):
    """
    Takes a GFFLineInfo instance and returns a feature-specific object

    """
    if feat_type is None:
        feat_type = line_info.feat_type.lower()
    containers = {
        "gene": Gene,
        "transcript": Transcript,
        "exon": Exon,
        "cds": Exon
    }
    # default to Transcript class if it's not an obvious feature
    if feat_type not in containers:
        feat_type = "transcript"
    # Get standard feature info
    info = {
        "name": line_info.name,
        "feat_type": feat_type,
        "parent": line_info.parent,
        "region": line_info.region,
        "strand": line_info.strand,
        "start": line_info.start,
        "stop": line_info.stop,
        "line_number": line_info.line_number,
        "phase": line_info.phase
    }
    # Put each type of data in the right container
    C = containers[feat_type]  # Gene, Transcript, Exon
    # Initialize new instance
    new_feat = C(**info)  # dict unpacking FTW
    return new_feat


def has_feature(annotation, check_type, column=2):
    """
    Checks annotation file for presence of check_type
    in column position of each line

    Returns True at first match found, False if
    check_type not found on any line

    """
    with flex_open(annotation) as f:
        for l in f:
            if l.startswith("#"):
                continue
            bits = l.strip().split("\t")
            if len(bits) < 9:
                continue
            feat_type = bits[column].lower()
            if feat_type == check_type.lower():
                return True
    return False


def consolidate(*levels):
    """
    Arranges list of object dictionaries levels into a
    hierarchical data structure, based upon the attribute
    "parent"

    Must pass arguments in ascending order (e.g. children,
    parents, grandparents)

    """
    # consolidated = list(levels)  # modify copy
    consolidated = levels
    for i, lvl in enumerate(levels):
        try:
            parents = consolidated[i + 1]
        except IndexError:
            return consolidated[i]
        for name, obj in lvl.items():
            p = obj.parent
            if p in parents:
                parents[p].children.append(obj)
                obj.grandparent = parents[p].parent

    return consolidated


# Build a hierarchy of objects to allow for object functions to operate
# correctly (won't work if everything is flattened)
def annotation_hierarchy(annotation_file, *child_feats):
    """
    Build an object heirarchy of gene/transcript/child_feats
    entries from an annotation file and chosen feature(s).
    Examines entire file in case features are not already
    grouped in hierarchical structure.

    child_feats may be multiple features, which will all be
    assigned as children of Transcript entries.


    Returns a list of aggregate objects, as well as a
    dictionary of all objects indexed by name.

    """
    genes = {}
    transcripts = {}
    children = defaultdict(list)
    destination = {"gene": genes, "transcript": transcripts}
    destination.update({c: children for c in child_feats})
    # to_collect = parent_types + list(child_feats)
    parent_containers = {"gene": Gene, "transcript": Transcript}
    parent_containers.update({c: Transcript for c in child_feats})
    # Track parents that don't meet criteria to avoid adding their children
    # to list of objects
    # # Keep record of unique coordinates to allow setting of dupe flag
    # # if dupe children are detected
    unique_coords = defaultdict(set)
    with flex_open(annotation_file) as f:
        # Track line numbers for tagging downstream objects with
        # their source lines
        parent_map = {}
        for ln, l in enumerate(f):
            try:
                line_info = GFFLineInfo(l, ln)
            except TypeError:  # not a proper annotation line
                continue
            new_feat = make_feat_instance(line_info)
            parent = new_feat.parent
            # check to ignore duplicate child entries
            if new_feat.feat_type not in child_feats:  # can be None
                # if new_feat.feat_type.lower() in ('cds', 'exon'):  # why?
                #     continue
                parent_map[new_feat.name] = line_info
                if parent is not None and parent not in parent_map:
                    parent_map[parent] = Gene(name=parent)
                continue
            elif parent:
                check_coords = (
                    new_feat.feat_type,
                    new_feat.start,
                    new_feat.stop)
                if check_coords not in unique_coords[parent]:
                    children[new_feat.compute_name()] = new_feat
                    unique_coords[parent].add(check_coords)

    parent_dest = transcripts
    grandparent_dest = genes

    # now make parent objs
    for name, child in children.items():
        parent = child.parent
        # grandparent = child.parent
        if parent not in parent_dest:
            try:
                parent_info = parent_map[parent]
                parent_obj = make_feat_instance(parent_info, 'transcript')
                grandparent = parent_obj.parent
                gp_info = parent_map[grandparent]
                gp_obj = make_feat_instance(gp_info, 'gene')
            except KeyError:  # there was no parent line in gff
                # without parent line, use transcript name for
                # both gene and transcript
                parent_obj = Transcript(name=parent)
                gp_obj = Gene(name=parent)
                grandparent = parent

            # make parent and grandparent objs in respective
            # containers
            parent_dest[parent] = parent_obj
            grandparent_dest[grandparent] = gp_obj

    # Now collapse everything down to topmost objects
    collected = [e for e in [children, transcripts, genes] if e]
    consolidated = consolidate(*collected)
    if not consolidated:
        write_log(
            '[!] ERROR: could not establish parent-child relationships '
            'among feature set. Check annotation file format. Exiting now.'
        )
        sys.exit()
    top_level_objs = list(consolidated.values())
    # generate sibling numbers
    for obj in top_level_objs:
        obj.set_family_size()
        obj.coding_length = obj.get_coding_length()

    return top_level_objs


#TODO finish this function to allow creation of transcript/gene info file
# def high_level_metadata(obj_hierarchy, file_name):
#     """
#     Gets metadata about high-level genome features (genes, transcripts).
#     Returns a generator of formatted info strings.

#     """
#     for obj in obj_hierarchy:


def flatten(obj_list, all_objs=None, feat_list=None):
    """
    Takes a list of top-level objects and creates a dictionary
    of all feature objects in the list, including children.

    If feat_list, only returns objects of with >feat_type<s
    present in >feat_list<.

    """
    if all_objs is None:
        all_objs = {}
    for obj in obj_list:
        try:
            all_objs.update(
                flatten(obj.children, all_objs, feat_list)) # oooh recursion
            if feat_list:
                if obj.feat_type not in feat_list:
                    continue
            all_objs[obj.name] = obj
        except AttributeError:
            if feat_list:
                if obj.feat_type not in feat_list:
                    continue
            all_objs[obj.name] = obj

    return all_objs


def collect_introns(objs, feat_type):
    """
    Creates a dictionary of all intron objects from a
    list of higher-level objects.

    """
    introns = defaultdict(list)
    total_count = 0
    for o in objs:
        new_introns = o.get_introns(feat_type)
        for i in new_introns:
            total_count += 1
            introns[i.region].append(i)

    return introns, total_count


def coord_overlap(coords, coord_list):
    """
    Check a list of integer coordinate tuples
    >coord_list< for overlap with the tuple
    >coords<. Returns the first overlapping
    tuple from >coord_list<, or False if no
    overlapping tuples found.

    """
    coord_list = sorted(coord_list)
    starts, stops = zip(*coord_list)
    c_start, c_stop = coords
    start_idx = bisect_left(stops, c_start)
    stop_idx = bisect_right(starts, c_stop)
    if start_idx != stop_idx:
        overlap_idx = min(start_idx, stop_idx)
        return coord_list[overlap_idx]
    else:
        return False

# def coord_overlap(coords, coord_list):
#     """
#     Check a list of integer coordinate tuples
#     >coord_list< for overlap with the tuple
#     >coords<. Returns the first overlapping
#     tuple from >coord_list<, or False if no
#     overlapping tuples found.

#     """
#     return next(
#         (c for c in coord_list if overlap_check(coords, c)), False)


def add_tags(
    intron, intron_index,
    longest_isoforms, allow_overlap=False, longest_only=True
):
    """
    Add >duplicate<, >overlap< and >longest_isoform< tags
    to an intron.

    """
    region_id = (intron.region, intron.strand)
    i_coords = tuple([getattr(intron, f) for f in ['start', 'stop']])
    region_idx = intron_index[region_id]
    seen_coords = list(region_idx.keys())
    # is it a duplicate?
    if i_coords not in region_idx:
        intron.duplicate = False
        region_idx[i_coords]['max'] = intron.parent_length
        # region_idx[i_coords]['index'] = index
        region_idx[i_coords]['fam'] = intron.family_size
        region_idx[i_coords]['unique_num'] = intron.unique_num
    else:
        intron.duplicate = region_idx[i_coords]['unique_num']
        intron.overlap = intron.duplicate
    # is it in longest isoform?
    parent = intron.parent
    gene = intron.grandparent
    if gene not in longest_isoforms:
        longest_isoforms[gene] = intron.parent
        intron.longest_isoform = True
    elif intron.duplicate or longest_isoforms[gene] != parent:
        intron.longest_isoform = False
    else:
        intron.longest_isoform = True

    # only check for overlap if intron is not a duplicate
    if intron.duplicate is False:
        if (intron.longest_isoform is not True and
            allow_overlap is False and longest_only is False):
            # iff the intron isn't in the longest isoform, check if its
            # coords overlap any other intron's coords and tag it
            # accordingly
            overlap = coord_overlap(i_coords, seen_coords)
            if overlap:
                overlap = region_idx[overlap]['unique_num']
                intron.overlap = overlap
            else:
                intron.overlap = False

    return intron, intron_index, longest_isoforms


def intronator(exons):
    """
    Builds introns from pairs of Exon objects
    in >exons<.

    sort_attr is used for ordering of introns.

    """

    def _window(seq):
        """
        Taken from https://docs.python.org/release/2.3.5/lib/itertools-example.html

        Returns a sliding window (of width n) over data from the iterable
        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        """
        n = 2
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    exons = sorted_in_coding_direction(exons)
    # accumulate lengths of children to calculate intron phases
    # and relative positions at each index
    exon_lengths = [e.length for e in exons]
    child_cumsum = np.array(exon_lengths)[:-1].cumsum()
    frac_positions = ((child_cumsum / sum(exon_lengths)) * 100).round(3)
    for index, pair in enumerate(_window(exons)):
        # edge case where exons might overlap in same transcript/gene;
        # causes problems downstream but parsing later requires more
        # work and it's rare enough that warning user should suffice
        pair_coords = [(p.start, p.stop) for p in pair]
        if overlap_check(*pair_coords):
            overlap_log(pair)
            # don't create intron from overlapping features
            continue
        new_intron = Intron.from_exon_pair(*pair)
        # Record up- and downstream exon names in case needed
        # for annotation file modification later
        us_ex, ds_ex = [ex.name for ex in pair]
        new_intron.upstream_exon = us_ex
        new_intron.downstream_exon = ds_ex
        new_intron.fractional_position = frac_positions[index]

        yield new_intron


def overlap_log(objs):
    parent = objs[0].parent
    coords = [(o.start, o.stop) for o in objs]
    write_log(
        '[!] WARNING: overlapping {} features found in {}: {} - skipping',
        FEATURE, parent, coords)


# def overlap_check(a, b):
#    """
#    Check to see if the two ordered tuples, >a< and >b<,
#    overlap with one another.

#    By way of Jacob Stanley <3
#    """

#    val = (a[0] - b[1]) * (a[1] - b[0])

#    if val < 0:
#        return True
#    else:
#        return False


def overlap_check(a, b):
    """
    >a< and >b< are sorted coordinate tuples;
    returns True if they overlap, False otherwise

    """
    lowest = min([a, b], key=lambda x: min(x))
    highest = max([a, b], key=lambda x: min(x))
    if min(highest) <= max(lowest):
        return True
    else:
        return False


def sorted_in_coding_direction(obj_list):
    """
    Sorts a list of GenomeFeature objects by their
    stop/start attribute, depending on the value of
    their strand attributes.

    """
    strands = set([o.strand for o in obj_list])
    if len(strands) > 1:  # can't pick single orientation
        write_log(
            "WARNING: mixed strands found in provided object list "
            "(first object name: '{}'); "
            "defaulting to + strand orientation", obj_list[0].name)
        # default to sorting for + strand if mixed strands
        return sorted(obj_list, key=attrgetter('start'))
    else:
        strand = strands.pop()
        if strand == "-":
            # Want to sort by "first" coord, which for reverse-
            # strand features is the stop
            coord = "stop"
            rev = True
        else:
            coord = "start"
            rev = False
        return sorted(obj_list, key=attrgetter(coord), reverse=rev)


def sliding_window(seq, n):
    """
    Taken from https://docs.python.org/release/2.3.5/lib/itertools-example.html

    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def seq_score(seq, matrix, start_index=0):
    """
    Score >seq< using values from >matrix<.

    Returns a float.

    """
    score = None
    for i, e in enumerate(seq, start=start_index):
        if score is None:
            score = matrix[e][i]
        else:
            score *= matrix[e][i]

    return score


# def bp_score(seq, matrix):
#     """
#     Score every sub-sequence of >seq< with length equal
#     to the value of the keys in >matrix<.

#     Returns the highest score achieved by any sub-sequence,
#     the relative coords of that sub-sequence within >seq<,
#     and the sub-sequence itself.

#     """
#     # If the matrix has different lengths for the value of any key,
#     # use the shortest
#     window_size = matrix_length(matrix)
#     start = 0
#     stop = window_size
#     best_score = None
#     best_coords = None
#     best_seq = None
#     for sub_seq in sliding_window(seq, window_size):
#         # convert from tuple to string
#         sub_seq = ''.join(sub_seq)
#         new_score = seq_score(sub_seq, matrix)
#         new_coords = (start, stop)
#         if best_score is None:
#             best_score = new_score
#             best_coords = new_coords
#             best_seq = sub_seq
#         else:
#             if new_score > best_score:
#                 best_score = new_score
#                 best_coords = new_coords
#                 best_seq = sub_seq
#         # Adjust window coordinates for next round
#         start += 1
#         stop += 1
#     return best_score, best_coords, best_seq


# bp score version with position-specific score weighting
# for Physarum
def bp_score(seq, matrix, use_bpx=False):
    """
    Score every sub-sequence of >seq< with length equal
    to the value of the keys in >matrix<.

    Returns the highest score achieved by any sub-sequence,
    the relative coords of that sub-sequence within >seq<,
    and the sub-sequence itself.

    """
    # If the matrix has different lengths for the value of any key,
    # use the shortest
    window_size = matrix_length(matrix)
    start = 0
    stop = window_size
    best_score = None
    best_coords = None
    best_seq = None
    best_bpx_mod = None
    for sub_seq in sliding_window(seq, window_size):
        # convert from tuple to string
        sub_seq = ''.join(sub_seq)
        if not valid_chars(sub_seq):
            start += 1
            stop += 1
            continue
        # flag to send back if branch point score was modified
        # using BPX
        bpx_mod = None
        new_score = seq_score(sub_seq, matrix)
        # calculate the distance of the end of the motif from
        # the 3' end of the full intron using the location of
        # the end of the bp region and the window's current
        # end coordinate
        if BPX and use_bpx is True:
            dist_from_3 = (len(seq) - stop) + abs(BP_REGION_COORDS[1])
            # perform multiplier if present in dictionary
            multiplier = BPX.get(dist_from_3)
            if multiplier is not None:
                # bpx_mod = ( (multiplier - 1) / 2 ) + 1
                bpx_mod = multiplier
                # bp score adjustment accounting for neg. initial scores
                delta = abs(new_score * (bpx_mod - 1))
                new_score += delta
        new_coords = (start, stop)
        if best_score is None:
            best_bpx_mod = bpx_mod
            best_score = new_score
            best_coords = new_coords
            best_seq = sub_seq
        elif new_score > best_score:
            best_bpx_mod = bpx_mod
            best_score = new_score
            best_coords = new_coords
            best_seq = sub_seq
        # Adjust window coordinates for next round
        start += 1
        stop += 1

    return best_score, best_coords, best_bpx_mod, best_seq


def matrix_from_seqs(seqs, start_index=0):
    """
    Constructs a position-weight matrix from one or
    more sequences.

    Returns a dictionary of {character: [f1, f2...fn]},
    where fn is the character frequency at the nth
    position in the sequence.

    """
    matrix = defaultdict(list)
    characters = set(['G', 'T', 'A', 'C'])
    lengths = set()
    n_seqs = 0

    for s in seqs:
        lengths.add(len(s))
        n_seqs += 1
        for i, e in enumerate(s, start=start_index):
            matrix[i].append(e)
            characters.add(e)
    
    if n_seqs == 0:
        return {}, 0

    character_order = sorted(characters)
    seq_length = min(lengths)

    frequencies = defaultdict(dict)

    for i in range(start_index, seq_length - abs(start_index)):
        chars = matrix[i]
        freqs = []
        for c in character_order:
            n = chars.count(c)
            f = n / n_seqs
            freqs.append(f)
            frequencies[c][i] = f

    return frequencies, n_seqs


def valid_chars(seq, bad_chars=re.compile("[^ACTG]")):
    """
    Returns True if >seq< contains only characters in the set
    [ACTG], False otherwise.

    """
    if bad_chars.search(seq):
        return False
    else:
        return True


def canonical_bounds(intron):
    """
    Checks an intron sequence's dinucleotide boundaries
    for canonical splice sites. Returns False if boundaries
    are non-canonical.

    Splice site variants derived from table 1 in
    DOI: 10.1093/nar/gku744

    """
    canonical = {
        "AT": ["AC"], #"AC", "AG", "AA", "AT"
        "GT": ["AG"], #"GG", "AA", "TG", "AT"
        "GC": ["AG"],
        # "GG": ["AG"],
        # "GA": ["AG"],
        # "TT": ["AG"]
    }
    # Find seq dinucleotides
    five, three = intron.dnts

    if five not in canonical:
        return False
    elif three not in canonical[five]:
        return False
    else:
        return True


def u12_correction(intron):
    """
    Checks >intron< for the presence of misannotated
    U12 boundaries. This requires a shift by the same number
    of nucleotides on both ends of the intron.

    If a correction can be made to produce a strong 5' boundary,
    the integer value of the shift will be placed in the intron's
    >corrected< attribute. Otherwise, >intron< is returned unchanged.

    """
    def _shift_phase(phase, shift):
        phases = deque([0, 1, 2])
        try:
            index = phases.index(int(phase))
        except ValueError:  # e.g. '.' for exons
            return phase
        phases.rotate(-shift)

        return phases[index]

    up_n = 5
    down_n = 12
    strict_motif = re.compile(r'[AG]TATC[CT]{2}')
    lax_motif = re.compile(r'[AG]TATC[CT]')
    # relax constraints if we're correcting a non-canonical intron
    # if canonical_bounds(intron):
    #     motif = strict_motif
    # else:
    #     motif = lax_motif
    motif = strict_motif
    region = intron.upstream_flank[-up_n:] + intron.seq[:down_n]
    match = motif.search(region)
    if not match:
        return False
    match_index = match.start()
    shift = match_index - up_n
    if shift == 0:
        return False
    intron.corrected = shift
    intron.phase = _shift_phase(intron.phase, shift)
    intron.dynamic_tag.add('[c:{}]'.format(shift))
    if intron.strand == '-':  # reverse adjustment for neg. strand
        shift *= -1
    intron.start += shift
    intron.stop += shift

    return True


def correct_annotation(introns, flattened, annotation_file, run_dir):
    """
    Adjusts intron-defining entries in >annotation_file<
    based upon the intron attribute >corrected<.

    If an intron's coordinates can be corrected with a -2 shift,
    for example, this would require shifting the stop coord
    of the 5' exon and the start coord of the 3' exon by -2 each.

    A ';shifted:[start, stop]:[1, 2, -1, -2]' tag is added to each
    line where such adjustments are made.

    >flattened< is a dictionary index of all objects in the annotation
    file indexed by name.

    Returns the modified filename.

    """

    # two functions to correct the phase of a sequence
    # whose start coordinate is changed by >shift< number
    # of nt
    def _shift_phase(phase, shift):
        phases = deque([0, 1, 2])
        try:
            index = phases.index(int(phase))
        except ValueError:  # e.g. '.' for exons
            return phase
        phases.rotate(-shift)

        return phases[index]

    # def phase_adjust(phase, shift):
    #     phases = [0, 1, 2]
    #     adjusted = phase + shift
    #     if not 0 <= adjusted <= 2:
    #         if adjusted < 0:
    #             adjusted = 3 + adjusted
    #         else:
    #             adjusted = abs(3 - adjusted)

    #     return phases[adjusted]


    def _correct_exon(exon, shift, side):
        """
        Corrects one of an exon object's coordinates,
        by amount >shift< and value of >side< (either
        5 or 3).

        Intended to function in the service of correcting
        exons to recapitulate canonical intron boundaries.

        """
        # Coordinate choice depends on which side of intron it's on
        if side == 5:
            target = 'stop'
            phase_shift = False
        else:
            target = 'start'
            # only flag for phase correction if start coord is modified
            phase_shift = True
        # Change goes the other way for negative strand coordinates
        if exon.strand == '-':
            shift = shift * -1
            # coords are always relative to + strand
            target = next(e for e in ('start', 'stop') if e != target)
        old_coord = getattr(exon, target)
        new_coord = old_coord + shift
        setattr(exon, target, new_coord)
        # Return extra values for use in _change_coords
        return exon, target, shift, phase_shift

    def _change_coords(line, coords, coord_tag, shift_tag, phase_shift):
        bits = line.strip().split('\t')
        insert = ';'
        if bits[8].endswith(';'):
            insert = ''
        mod_tag = '{}shift:{}:{}'.format(insert, coord_tag, shift_tag)
        bits[8] = bits[8] + mod_tag
        bits[3], bits[4] = map(str, coords)
        # correct phase if applicable
        if phase_shift is True:
            if bits[6] == '+':
                shift_tag *= -1
            # bits[7] = str(int(bits[7]) + (shift_tag * -1))
            bits[7] = str(_shift_phase(bits[7], shift_tag))
        return '\t'.join(bits)

    # Iterate over only those introns with non-zero values for >corrected<
    corrected_count = 0
    corrected_dupes = 0
    corrected_exon_coords = {}
    for intron in filter(attrgetter('corrected'), introns):
        corrected_count += 1
        if intron.duplicate:
            corrected_dupes += 1
        shift = intron.corrected
        five_exon = flattened[intron.upstream_exon]
        three_exon = flattened[intron.downstream_exon]
        # because exons are in coding orientation already,
        # we need to flip if on neg. strand to make the
        # coord adjustment work correctly
        ###!!!
        # if intron.strand == '-':
        #     five_exon, three_exon = three_exon, five_exon
        ###!!!
        # Build index of corrected coords by line number
        for ex, intron_side in zip([five_exon, three_exon], [5, 3]):
            (cor_ex, coord_tag,
            shift_tag, phase_shift) = _correct_exon(ex, shift, intron_side)
            coords = (cor_ex.start, cor_ex.stop)
            corrected_exon_coords[ex.line_number] = (
                coords, coord_tag, shift_tag, phase_shift)

    # We now have an index of lines to modify in the annotation file
    if corrected_count == 0:
        return
    modded_filename = FN_ANNOT
    modded_filepath = os.path.join(
        RUN_DIR, modded_filename)
    with flex_open(annotation_file) as infile, \
    open(modded_filepath, 'w') as outfile:
        for ln, l in enumerate(infile):
            if ln not in corrected_exon_coords:
                outfile.write(l)
                continue
            (new_coords, coord_tag,
            shift_tag, phase_shift) = corrected_exon_coords[ln]
            new_line = _change_coords(
                l, new_coords, coord_tag, shift_tag, phase_shift)
            outfile.write(new_line + '\n')
    write_log(
        '{} ({} unique, {} redundant) putatively misannotated U12 introns '
        'corrected in {}',
        corrected_count,
        corrected_count - corrected_dupes,
        corrected_dupes,
        modded_filename
    )


def write_log(string, *variables, wrap_chars='[]', level='info'):
    """
    Prints to screen and writes to log file a
    formatted string with >variables< surrounded
    by >wrap_chars<.

    e.g. write_log('The answer is {}', 42) --> 'The answer is [42]'

    If >wrap_chars< is None, variables are printed without
    additional formatting.

    """
    if wrap_chars is None:
        formatted_vars = variables
    else:
        open_char, close_char = wrap_chars
        formatted_vars = [
            '{}{}{}'.format(open_char, v, close_char) for v in variables
        ]
    formatted_string = string.format(*formatted_vars)
    logger = getattr(logging, level)
    logger(formatted_string)


def assign_seqs(
    intron,
    region_seq,
    int_flank_size,
    # five_flank,
    # three_flank,
    five_score_coords,
    three_score_coords,
    bp_coords
):
    """
    Assign sub-sequences to intron objects based on object metadata
    and provided variables in argument.

    Returns a modified intron object.

    """
    def _short_bp_adjust(intron, bpc, fsl):
        """
        Check for and adjust bp coords to correct
        for smaller initial bp search region in
        short introns (where original bp region coords
        might extend upstream of the 5' end of the intron)

        """
        ic = (intron.start, intron.stop)
        rev = False
        if intron.strand == '-':
            rev = True
            ic = ic[::-1]
            bpc = bpc[::-1]
            fsl *= -1
        bp_start, bp_stop = bpc
        int_start = ic[0]
        shift = abs((int_start + fsl) - bp_start)
        if rev:
            bp_start -= shift
            bpc = (bp_stop, bp_start)
        else:
            bp_start += shift
            bpc = (bp_start, bp_stop)
        return bpc

    upf, intron.seq, downf = intron.get_seq(region_seq, flank=int_flank_size)
    intron.upstream_flank = upf
    intron.downstream_flank = downf

    # scoring sequences
    us_length = len(intron.upstream_flank)
    ds_length = len(intron.downstream_flank)

    # adjust coords to account for flanking sequence
    scoring_seq = upf + intron.seq + downf
    five_rel_coords = [c + us_length for c in five_score_coords]
    three_rel_coords = [c - ds_length for c in three_score_coords]

    # remove zeroes to avoid indexing errors if range ends in 0
    five_rel_coords = [c if c != 0 else None for c in five_rel_coords]
    three_rel_coords = [c if c != 0 else None for c in three_rel_coords]

    # pull sequence corresponding to relative coordinate ranges
    intron.five_seq = scoring_seq[slice(*five_rel_coords)]
    intron.three_seq = scoring_seq[slice(*three_rel_coords)]

    # fixed five and three sequences for display purposes
    intron.three_display_seq = intron.seq[bp_coords[1]:]
    intron.five_display_seq = intron.seq[:10]
    intron.dnts = (intron.seq[:2], intron.seq[-2:])
    if not canonical_bounds(intron):
        intron.noncanonical = True
    else:
        intron.noncanonical = False

    # account for the 1-based indexing adjustment in get_seqs()
    # which should not apply to these kinds of relative coords
    bp_coords = (bp_coords[0] + 1, bp_coords[1])
    bp_region_coords = intron.get_rel_coords('three', bp_coords)

    # correct bp region if intron is short
    five_score_length = len([e for e in range(*five_score_coords) if e >=0])
    if intron.length < abs(bp_coords[0]) + five_score_length:
        bp_region_coords = _short_bp_adjust(
            intron, bp_region_coords, five_score_length)

    intron.bp_region_seq = intron.get_seq(region_seq, *bp_region_coords)

    return intron


def longest_match(seq, pattern=r'[ATCG]+'):
    """
    Takes a string, and returns the length of
    the longest stretch of characters matching
    {pattern}.

    """
    matches = re.findall(pattern, seq)
    if matches:
        return len(max(matches, key=len))
    else:
        return 0


def get_sub_seqs(
    introns_by_region,
    flat_objs,
    genome,
    int_flank_size,
    # five_flank,
    # five_score_length,
    five_score_coords,
    three_score_coords,
    bp_coords
):
    """
    Generator that populates objects in >introns< with short
    sub-sequences using >genome<.

    >introns_by_region< is a dictionary of intron objects keyed
    with the >region< attribute.

    """
    # get the total number of FASTA headers we need to consider
    # to allow for early exit if there are a bunch of headers
    # without introns
    total_regions = len(introns_by_region)
    intron_index = defaultdict(lambda: defaultdict(dict))
    for region_name, region_seq in fasta_parse(genome):
        if region_name not in introns_by_region:
            continue
        longest_isoforms = {}
        region_seq = region_seq.upper()
        for intron in sorted(
            introns_by_region[region_name],
            key=lambda i: (
                i.parent_length,
                i.family_size,
                -1 * i.line_number), reverse=True):
            intron = assign_seqs(
                intron,
                region_seq,
                int_flank_size,
                # five_flank,
                # five_score_length,
                five_score_coords,
                three_score_coords,
                bp_coords)
            if intron.noncanonical:
                if u12_correction(intron):  # coords have changed
                    intron = assign_seqs(
                        intron,
                        region_seq,
                        int_flank_size,
                        # five_flank,
                        # five_score_length,
                        five_score_coords,
                        three_score_coords,
                        bp_coords)
            # # add all intron tags
            # intron, intron_index, longest_isoforms = add_tags(
            #     intron, flat_objs, intron_index, longest_isoforms,
            #     ALLOW_OVERLAP)

            yield intron

        total_regions -= 1
        if total_regions == 0:  # we've used all the headers we need
            break


def write_format(obj, *attribs, fasta=True, separator='\t', null='.'):
    """
    Formats a set of object attributes into a string
    for writing to a file.

    If >fasta< is True, will use the first attribute
    as a header in FASTA format.

    Can retrieve method attributes, but not ones which
    require arguments.

    No trailing newline in returned string.

    """
    attribs = list(attribs)
    values = []
    for atr in attribs:
        try:
            value = getattr(obj, atr)
        except AttributeError:  # is simply a variable being passed in
            values.append(atr)
            continue
            # try them as functions; if that doesn't work, they're
            # already values
        try:
            value = value()
        except TypeError:
            pass
        values.append(value)
    if fasta is True:
        header = '>{}'.format(values.pop(0))
    if null is not None:
        values = [v if v is not None else null for v in values]
    content = separator.join([str(v) for v in values])
    if fasta is True:
        content = '\n'.join([header, content])

    return content


def counter_format_top(cntr, num=None):
    """
    Formats a Counter object >cntr< for printing its
    values in a numbered list, along with percentage
    statistics for each value.

    >num< is the number of entries that will be printed.
    If >num< is None, all values will be printed.

    Returns an iterator of formatted strings.

    """
    top = cntr.most_common(num)
    total = sum(cntr.values())
    for element, count in top:
        fraction = (count / total) * 100
        count_info = "* {} ({}/{}, {:.2f}%)".format(
            element, count, total, fraction)
        yield count_info


def build_u2_bp_matrix(introns, u12_matrix, dnt_list=None):
    """
    Builds a matrix for branch point sequences
    based on highest-scoring sequences when
    scored using a u12 bp matrix.

    Returns a matrix dictionary.

    """
    def _iter_bps(ints, matrices, dnt_list=None):
        bp_seqs = []
        for intron in ints:
            if dnt_list is not None:
                if intron.dnts not in dnt_list:
                    continue
            bp_region_seq = intron.bp_region_seq
            
            ###!!!
            best_score = 0
            best_seq = None
            for matrix in matrices:
                m_score, *_, seq = bp_score(bp_region_seq, matrix, use_bpx=True)
                if m_score > best_score:
                    best_seq = seq
                    best_score = m_score
            seq = best_seq
            ###!!!

            if not seq:
                print('NO BP SEQ: ', intron.get_name())
                sys.exit(0)
            yield seq

    bp_seqs = _iter_bps(introns, u12_matrix, dnt_list)

    return matrix_from_seqs(bp_seqs)


def matrix_length(matrix):
    """
    Returns the shortest value length found in >matrix<

    """
    return min(set([len(vals) for vals in matrix.values()]))


def get_score_bounds(matrix):
    """
    Retrieves the highest and lowest possible scores for >matrix<,
    e.g. the score achieved if a sequence matched the highest-
    or lowest-frequency character at every position in the matrix.

    """
    position_indexed = defaultdict(list)
    # Collect frequencies at each position in lists
    for character, freqs in matrix.items():
        for i, f in freqs.items():
            position_indexed[i].append(f)
    # Calculate max and min values for each position
    min_freqs, max_freqs = [], []
    for position, freqs in position_indexed.items():
        min_freqs.append(min(freqs))
        max_freqs.append(max(freqs))

    # Total scores are products of each list
    min_score = np.prod(min_freqs)
    max_score = np.prod(max_freqs)

    return min_score, max_score


def multi_matrix_score(
    intron,
    matrices,
    regions=('five', 'bp', 'three'),
    matrix_tags=None,
    use_bpx=False):
    """
    Finds the highest-scoring matrix key and value for the
    specified intron, using the sum of multiple regions for
    scoring.

    Returns a tuple of the score and associated matrix key.

    """
    region_map = {
        'five': 'five_seq',
        'bp': 'bp_region_seq',
        'three': 'three_seq'
    }
    score_funcs = {
        'five': partial(seq_score, start_index=FIVE_SCORE_COORDS[0]),
        'bp': partial(bp_score, use_bpx=use_bpx),
        'three': partial(seq_score, start_index=THREE_SCORE_COORDS[0])}
    score_info = defaultdict(lambda: defaultdict(dict))
    if matrix_tags is not None:
        matrices = {
            k: v for k, v in matrices.items()
            if all(t in k for t in matrix_tags)}
    else:
        matrices = {
            k: v for k, v in matrices.items()
            if any(r in k for r in regions)}
    for matrix_key, matrix in matrices.items():
        subtype, dnts, region, *_ = matrix_key
        matrix_category = (subtype, dnts)
        score_function = score_funcs[region]
        seq = getattr(intron, region_map[region])
        score = score_function(seq, matrix)
        if score is None:  # edge case where sequence is at end of contig
            score = PSEUDOCOUNT * len(matrix)
        if region == 'bp':
            # bp score function returns additional information
            s = score[0]
            info = score[1:]
        else:
            s = score
            info = []
        score_info[matrix_category][region]['score'] = s
        score_info[matrix_category][region]['info'] = info

    return score_info


def best_matrix(score_info, scoring_regions, priority_tag=None):
    # summary_score_regions = ['five', 'bp', 'three']
    category_scores = []
    for matrix_category, regions in score_info.items():
        region_scores = []
        for r, r_score in regions.items():
            score = r_score['score']
            if r in scoring_regions:
                region_scores.append(score)
        summary_score = summarize(region_scores)
        category_scores.append((summary_score, matrix_category))

    # filter best scores by tag if present, otherwise simply return
    # highest-scoring matrix
    category_scores.sort(reverse=True)
    if priority_tag:
        try:
            best_score_info = next(
                s for s in category_scores if priority_tag in s[1])
        except StopIteration:
            best_score_info = category_scores[0]
    else:
        best_score_info = category_scores[0]
    # Return top score and matrix category
    return best_score_info  # e.g. (score, (u12, gtag))


def log_ratio(a, b):
    """
    Computes the log base 2 score of a over b.

    """
    return math.log2(a / b)


def get_raw_scores(introns, matrices, multiplier_d=None):
    """
    Assigns scores to each intron in >introns<,
    using available matrices in >matrices<.

    Returns a generator of introns with raw scores added

    """
    for intron in introns:
        # Determine the best type of matrix to used based on 5' score
        # e.g. ('u12', 'gtag')
        u2_matrix_info = multi_matrix_score(
            intron, matrices, matrix_tags=['u2'])
        u12_matrix_info = multi_matrix_score(
            intron, matrices, matrix_tags=['u12'], use_bpx=True)
        dnts = ''.join(intron.dnts).lower()
        u2_score, best_u2_key = best_matrix(
            u2_matrix_info, SCORING_REGIONS, dnts)
        u12_score, best_u12_key = best_matrix(
            u12_matrix_info, SCORING_REGIONS, dnts)
        # u2_bp_k = ('u2', 'bp', 'gtag') # always use this matrix for u2
        # u2_bp_score, *_ = bp_score(
        #     intron.bp_region_seq,
        #     matrices[u2_bp_k])
        # u12_bp_k = (best_u12_key[0], 'bp', best_u12_key[2])
        # u12_bp_score, bp_rel_coords, bpm, u12_bp_seq = bp_score(
        #     intron.bp_region_seq,
        #     matrices[u12_bp_k], use_bpx=True)
        u12_bp_score = u12_matrix_info[best_u12_key]['bp']['score']
        u12_bp_info = u12_matrix_info[best_u12_key]['bp']['info']
        bp_rel_coords, bpm, u12_bp_seq = u12_bp_info

        if bpm is not None:
            intron.dynamic_tag.add('bpm={}'.format(bpm))

        # Get log scores for each region
        for region, attr in zip(
            ['five', 'bp', 'three'],
            ['five_raw_score', 'bp_raw_score', 'three_raw_score']
        ):
            ratio_score = log_ratio(
                u12_matrix_info[best_u12_key][region]['score'],
                u2_matrix_info[best_u2_key][region]['score'])
            setattr(intron, attr, ratio_score)

        intron.bp_seq = u12_bp_seq
        intron.bp_relative_coords = bp_rel_coords

        # record the name of the matrix used for each score type
        intron.u2_matrix = best_u2_key  # e.g. ('u12', 'gtag')
        intron.u12_matrix = best_u12_key

        yield intron


def unique_attributes(introns, a_list):
    seen_attrs = set()
    for i in introns:
        i_attrs = get_attributes(i, a_list)
        for ia in i_attrs:
            if ia not in seen_attrs:
                seen_attrs.add(ia)
                yield i


def make_z_score(raw_score, mean, stdev):
    try:
        z_score = (raw_score - mean) / stdev
        return z_score
    except RuntimeWarning:  # if stdev = 0
        return raw_score - mean


def get_raw_score_stats(introns, sample=False):
    # Store raw scores in lists, to be converted to arrays
    fives, bps, threes = [], [], []
    for intron in introns:
        fives.append(intron.five_raw_score)
        bps.append(intron.bp_raw_score)
        threes.append(intron.three_raw_score)

    stats = {
        'five': {'mean': 0, 'stdev': 0},
        'bp': {'mean': 0, 'stdev': 0},
        'three': {'mean': 0, 'stdev': 0}
    }
    fives = np.array(fives)
    bps = np.array(bps)
    threes = np.array(threes)
    for score_region, score_array in zip(
        ['five', 'bp', 'three'],
        [fives, bps, threes]
    ):
        stats[score_region]['mean'] = np.mean(score_array)
        stats[score_region]['bounds'] = (score_array.min(), score_array.max())
        # numpy stdev calculation is not corrected for sample by default
        # unclear if should correct for sample (done with >ddof<)
        # see:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
        try:
            stats[score_region]['stdev'] = np.std(score_array, ddof=sample)
        except RuntimeWarning:  # single intron in set
            stats[score_region]['stdev'] = 0

    return stats


def adjust_scores(introns, log=True, sample=False, stats=None):
    """
    Takes the raw scores of introns and adjusts them to
    Z scores by subtracting the mean and dividing by the
    standard deviation.

    If >stats<, will use provided statistics to calculate
    scores.

    Returns an iterator of introns with z scores.

    """
    def _adj_scores(introns_to_adjust):
    # Calculate z scores for each intron
        attribs_to_adjust = list(zip(
                    ['five', 'bp', 'three'],
                    ['five_raw_score', 'bp_raw_score', 'three_raw_score'],
                    ['five_z_score', 'bp_z_score', 'three_z_score']
        ))
        for intron in introns_to_adjust:
            for region, getattrib, setattrib in attribs_to_adjust:
                raw_score = getattr(intron, getattrib)
                z_score = make_z_score(
                    raw_score,
                    stats[region]['mean'],
                    stats[region]['stdev']
                )
                setattr(intron, setattrib, z_score)

            # Intron now has raw scores and z scores for both regions
            yield intron

    # Correct for sample in statistics or not
    if sample is False:
        sample = 0
    else:
        sample = 1

    # Build new list because >introns< is iterator
    # introns = [i for i in introns]

    # Do not build stats specific to the intron set if others
    # were provided
    if not stats:  # get stats from passed data
        stats = get_raw_score_stats(introns, sample=sample)

    if log:
        for print_name, key in zip(
            ['5\'', 'branch-point', '3\''],
            ['five', 'bp', 'three']):
            write_log(
                '{}: mean = {}, standard deviation = {}',
                print_name,
                stats[key]['mean'],
                stats[key]['stdev']
            )

    return stats, _adj_scores(introns)


def mutate(intron, five_score_coords, three_score_coords):
    intron.dnts = ('GT', 'AG')
    five_before = intron.five_seq[:abs(five_score_coords[0])]
    five_after = intron.five_seq[abs(five_score_coords[0]) + 2:]
    intron.five_seq = five_before + 'GT' + five_after
    three_before = intron.three_seq[:abs(three_score_coords[0]) - 2]
    three_after = intron.three_seq[abs(three_score_coords[0]):]
    intron.three_seq = three_before + 'AG' + three_after

    return intron


def write_matrix_file(matrices, fn):
    """
    Writes one or more matrices to a file, using
    keys as headers.

    """
    with open(fn, 'w') as outfile:
        for name, matrix in sorted(matrices.items()):
            label = '-'.join(name)
            # outfile.write('[#] {}\n'.format('-'.join(name)))
            outfile.write(format_matrix(matrix, label) + '\n')


def min_max_from_bounds(u12_bounds, u2_bounds):
    u12_min, u12_max = u12_bounds
    u2_min, u2_max = u2_bounds

    min_score = log_ratio(u12_min, u2_max)
    max_score = log_ratio(u12_max, u2_min)

    return min_score, max_score


def rescale(old_score, oldmin, oldmax, newmin=1e-5, newmax=1):
    scaled_score = (
        ((old_score - oldmin) * (newmax - newmin)) /
        ((oldmax - oldmin)) + newmin)

    return scaled_score


def annotate(string, start, stop):
    annotated = string[:start] + '[' + string[start:stop] + ']' + string[stop:]
    return annotated


def ranking_format(intron, ref=False, index=False):
    """
    Formats intron info for output in rankings file.
    Returns a string.

    """
    schematic = '{}|{}...{}...{}|{}'.format(
        intron.upstream_flank[-3:],
        intron.five_display_seq, intron.bp_seq,
        intron.three_display_seq, intron.downstream_flank[:3])
    bp_context = annotate(
        intron.bp_region_seq,
        *intron.bp_relative_coords) + intron.three_display_seq
    try:
        rounded_score = round(intron.relative_score, 4)
    except TypeError:  # for ref intron in log (deprecated)
        rounded_score = '-'
    rankings_bits = [
        index,
        schematic,
        rounded_score,
        bp_context,
        intron.length,
        intron.get_name()
    ]
    if ref:
        rankings_bits[-1] = intron.name
    formatted = '\t'.join([str(e) for e in rankings_bits if e is not None])

    return formatted


def dupe_list_format(intron, dupe_index):
    """
    Check an intron against an index of duplicates, and
    return a string of all duplicate entries in list format,
    or None.

    """
    intron_uid = (intron.region, intron.start, intron.stop)
    if intron_uid in dupe_index:
        list_bits = [intron.region, intron.strand, intron.start, intron.stop]
        dupe_strings = []
        for dupe in dupe_intron_index[intron_uid]:
            # dupe.svm_score = intron.svm_score
            dupe_name = dupe.get_label()
            dupe_bits = list_bits + [dupe_name]
            dupe_string = '\t'.join(map(str, dupe_bits))
            dupe_strings.append(dupe_string)
        dupe_output = '\n'.join(dupe_strings)
    else:
        dupe_output = None

    return dupe_output


def flip_check(intron, flip_dict):
    """
    Checks an intron against a dictionary of introns with scores
    that did not survive boundary switching.

    If present, the score resulting from the boundary switch
    will be returned, otherwise None.

    """
    name = intron.get_name()
    if name in flip_dict:
        return flip_dict[name]
    else:
        return None


def bp_multiplier(intron, multiplier_dict):
    """
    Checks the position of the branch point seq and adds a
    multiplier from multiplier_dict if the bp seq distance
    from the 3' end is in multiplier_dict.

    Returns None, or the appropriate multiplier.

    """
    bpr_length = len(intron.bp_region_seq)
    bp_stop = intron.bp_relative_coords[1]
    # Calculate distance from the 3' end of the intron
    d_from_3 = (bpr_length - bp_stop) + abs(BP_REGION_COORDS[1])

    return multiplier_dict.get(d_from_3)


def add_scores(introns, intron_seq_file):
    """
    Add scores to previously-made intron sequences file

    """
    # devlin
    devlin_file = open('../info/{}_info.iic'.format(SPECIES), 'w')
    #/ devlin

    tmp_file = '{}.tmp'.format(intron_seq_file)
    score_dict = {
        intron.name.split(';')[0]: intron.svm_score
        for intron in introns}
    with open(intron_seq_file) as inf, open(tmp_file, 'w') as outf:
        for l in inf:
            bits = l.strip().split('\t')
            # bits[0] is intron name
            name = bits[0].split(';')[0]
            if name in score_dict:
                score = str(round(score_dict[name], 3))
                bits[1] = score
            outf.write('\t'.join(bits) + '\n')
            
            # devlin
            try:
                devlin_bits = devlin_dict[name]
                seq = bits[4]
                devlin_bits.append(seq)
                devlin_file.write('\t'.join([str(e) for e in devlin_bits]) + '\n')
            except KeyError:
                pass

    devlin_file.close()
    #/ devlin

    os.remove(intron_seq_file)
    os.rename(tmp_file, intron_seq_file)


def summarize(scores):
    summary = pystats.gmean(scores)
    # summary = np.cbrt(np.prod(scores))
    # summary = np.mean(scores)
    # summary = np.sqrt(sum([s ** 2 for s in scores]))

    return summary


def summary_score(region_thresholds, weights=None):
    if weights is not None:
        region_thresholds = {r: t * weights[r] for r, t in region_thresholds.items()}

    return summarize(list(region_thresholds.values()))


def progress_bar(current, total, done_char='#', fill_char=' ', length=20):
    fraction_done = current / total
    n_done = math.floor(fraction_done * length)
    n_remaining = length - n_done
    prog_bar = (done_char * n_done) + (fill_char * n_remaining)
    prog_bar = '[{}]'.format(prog_bar)

    return prog_bar


def make_matrices(
    introns, 
    u12_threshold,
    five_start_coord,
    three_start_coord,
    regions=['five', 'bp'], 
    min_seqs=5):
    seq_attrs = {
        'five': 'five_seq',
        'bp': 'bp_seq',
        'three': 'three_seq'
    }
    region_starts = {
        'five': five_start_coord,
        'three': three_start_coord,
        'bp': 0
    }
    u12_atac = re.compile('^AT[ACT]{3}')
    u12_gtag = re.compile('^GT[ACT]{3}')
    u2_gtag = re.compile('^GT')
    u2_gcag = re.compile('^GC')
    u12_motifs = {
        ('u12', 'atac'): {
            'pattern': u12_atac,
            'dnts': ('AT', 'AC')},
        ('u12', 'gtag'): {
            'pattern': u12_gtag,
            'dnts': ('GT', 'AG')}
    }
    u2_motifs = {
        ('u2', 'gtag'): {
            'pattern': u2_gtag,
            'dnts': ('GT', 'AG')},
        ('u2','gcag'): {
            'pattern': u2_gcag,
            'dnts': ('GC', 'AG')}
    }
    # define generators to feed sequences
    matrices = {}
    for r in regions:
        attr_name = seq_attrs[r]
        attr_start = region_starts[r]
        for subtype, info in u12_motifs.items():
            subtype_key = subtype + (r,)
            subtype_motif = info['pattern']
            dnts = info['dnts']
            u12_seqs = (
                getattr(i, attr_name) for i in introns if 
                i.svm_score > u12_threshold and 
                i.dnts == dnts and
                subtype_motif.match(i.five_display_seq)
            )
            matrix, n_seqs = matrix_from_seqs(
                u12_seqs, start_index=attr_start)
            if n_seqs > min_seqs:
                matrices[subtype_key] = matrix
        for subtype, info in u2_motifs.items():
            subtype_key = subtype + (r,)
            subtype_motif = info['pattern']
            dnts = info['dnts']
            u2_seqs = (
                getattr(i, attr_name) for i in introns if 
                i.type_id == 'u2' and 
                i.dnts == dnts and
                subtype_motif.match(i.five_display_seq)
            )
            matrix, n_seqs = matrix_from_seqs(
                u2_seqs, start_index=attr_start)
            if n_seqs > min_seqs:
                matrices[subtype_key] = matrix
    
    return matrices


def add_u2_matrix(matrices, introns, min_u2_count=100):
    u2_bp_key = ('u2', 'gtag', 'bp')
    u12_five_key = ('u12', 'gtag', 'five')

    if u2_bp_key not in matrices:
        # get the 99th percentile of 5' scores to cull putative U12s
        five_scores = [
            seq_score(
                i.five_seq, 
                matrices[u12_five_key], 
                start_index=FIVE_SCORE_COORDS[0]
            )
            for i in introns
        ]
        u2_threshold = np.percentile(five_scores, 99)
        u2_bp_introns = (
            i for i, five in zip(introns, five_scores) 
            if five <= u2_threshold
        )
        # u2_bp_matrix, n_introns_used = build_u2_bp_matrix(
        #     u2_bp_introns,
        #     matrices[('u12', 'gtag', 'bp')], dnt_list=[('GT', 'AG'), ('GC', 'AG')])
        u12_bp_matrices = [
            v for k, v in matrices.items() if
            all(x in k for x in ['u12', 'bp', 'gtag'])]
        u2_bp_matrix, n_introns_used = build_u2_bp_matrix(
            u2_bp_introns,
            u12_bp_matrices, 
            dnt_list=[('GT', 'AG'), ('GC', 'AG')]
        )
        # use conserved u2 bp matrix as fallback if insufficient
        # u2s in provided data
        if n_introns_used < min_u2_count:
            u2_bp_matrix = load_external_matrix(U2_BPS_MATRIX_FILE)
            u2_bp_matrix = u2_bp_matrix[u2_bp_key]
            write_log(
                ('Insufficient U2 introns available to build BPS matrix; '
                 'using {} instead'), U2_BPS_MATRIX_FILE
            )
        else:
            write_log(
            '{} introns used to build U2 branch point matrix',
            n_introns_used)
        matrices[u2_bp_key] = add_pseudos(u2_bp_matrix, pseudo=PSEUDOCOUNT)
        matrices[('u2', 'gcag', 'bp')] = matrices[u2_bp_key]
    if ('u2', 'gcag', 'bp') not in matrices:
        matrices[('u2', 'gcag', 'bp')] = matrices[u2_bp_key]
        
    return matrices


def get_flipped(introns, model, scaler):
    flipped = {}
    swap_introns = copy.deepcopy(
        [
            i for i in introns if 
            (
                i.dnts == ('AT', 'AC') and not 
                i.five_display_seq.startswith('ATATC')
            )
            or i.noncanonical == True
        ]
    )
    # Only perform these calculation if we actually found applicable introns
    if swap_introns:
        raw_swaps = list(get_raw_scores(swap_introns, MATRICES))
        scored_swaps = scale_scores(raw_swaps, scaler)
        mutant_swaps = [
            mutate(intron, FIVE_SCORE_COORDS, THREE_SCORE_COORDS)
            for intron in copy.deepcopy(swap_introns)]
        raw_mutants = list(get_raw_scores(mutant_swaps, MATRICES))
        scored_mutants = scale_scores(raw_mutants, scaler)
        scored_swaps = assign_svm_scores(
            scored_swaps, model, scoring_region_labels)

        scored_mutants = assign_svm_scores(
            scored_mutants, model, scoring_region_labels)

        for old, new in zip(scored_swaps, scored_mutants):
            if old.svm_score > THRESHOLD and new.svm_score <= THRESHOLD:
                name = old.get_name()
                flipped[name] = new

    return flipped
    

def demote(introns, flipped):
    # For putative U12 introns whose scores don't survive dnt switching
    # demoted_swaps = []
    for i in introns:
        if (i.svm_score < THRESHOLD or (
            i.dnts != ('AT', 'AC') and i.noncanonical is False)):
            yield i
            continue
        if '[d]' in i.dynamic_tag:
            i.dynamic_tag.remove('[d]')
        flipped_i = flip_check(i, flipped)
        if flipped_i is not None:
            old_score = i.svm_score
            old_five = i.five_z_score
            old_bp = i.bp_z_score
            old_three = i.three_z_score
            flip_relative = flipped_i.relative_score
            flip_score = flipped_i.svm_score
            flip_five = flipped_i.five_z_score
            flip_bp = flipped_i.bp_z_score
            flip_three = flipped_i.three_z_score
            flip_dynamic_tag = flipped_i.dynamic_tag
            flip_type_id = flipped_i.type_id
            flip_info = [
                i.get_name(), old_score, old_five, old_bp, old_three,
                flip_score, flip_five, flip_bp, flip_three
            ]
            i.demote_info = flip_info
            # intron = flipped_intron   # produces incorrect display seqs

            # Set original intron's score to new adjusted score
            demote_attrs = {
                'svm_score': flip_score,
                'relative_score': flip_relative,
                'five_z_score': flip_five,
                'bp_z_score': flip_bp,
                'three_z_score': flip_three,
                'type_id': flip_type_id
            }
            for da, v in demote_attrs.items():
                setattr(i, da, v)

            i.dynamic_tag.add('[d]')

        yield i


def set_attributes(objs, attr_list, attr_names):
    new_objs = []
    for o, attrs in zip(objs, attr_list):
        for a, name in zip(attrs, attr_names):
            setattr(o, name, a)
        new_objs.append(o)
    
    return new_objs


def scale_scores(
    introns,
    scaler,
    get_names=['five_raw_score', 'bp_raw_score', 'three_raw_score'], 
    set_names=['five_z_score', 'bp_z_score', 'three_z_score']):
    s_vect = get_score_vector(
        introns, get_names)
    s_vect = scaler.transform(s_vect)
    introns = set_attributes(introns, s_vect, set_names)

    return introns


def apply_scores(
    ref_set, 
    exp_set, 
    matrices,
    scoring_regions, 
    log=True):
    # Get the raw log ratio scores of each scoring region in each intron
    raw_introns = get_raw_scores(exp_set, matrices)
    raw_introns = list(raw_introns)

    # Same for the reference sequences
    raw_refs = get_raw_scores(ref_set, matrices)
    raw_refs = list(raw_refs)

    raw_score_names = ['five_raw_score', 'bp_raw_score', 'three_raw_score']
    # scale everything together
    scale_vector = get_score_vector(
        raw_refs + raw_introns, score_names=raw_score_names)
    
    # make a scaler to adjust raw scores --> z-scores
    score_scaler = preprocessing.StandardScaler().fit(scale_vector)

    scored_refs = scale_scores(raw_refs, score_scaler)
    ref_u12s = [i for i in scored_refs if i.type_id == 'u12']
    ref_u2s = [i for i in scored_refs if i.type_id == 'u2']
    if log is True:
        write_log(
            'Raw scores calculated for {} U2 and {} U12 reference introns',
            len(ref_u2s), len(ref_u12s)
        )
    scored_introns = scale_scores(raw_introns, score_scaler)
    if log is True:
        write_log(
            'Raw scores calculated for {} experimental introns',
            len(scored_introns)
        )
    
    # make non-redundant score vectors for training data
    ref_u12_vector = get_score_vector(
        ref_u12s, score_names=scoring_regions)
    ref_u12_vector = np.unique(ref_u12_vector, axis=0)
    ref_u2_vector = get_score_vector(
        ref_u2s, score_names=scoring_regions)
    ref_u2_vector = np.unique(ref_u2_vector, axis=0)

    if log is True:
        write_log(
            'Non-redundant training set sizes: {} U2, {} ',
            len(ref_u2_vector), len(ref_u12_vector))
        write_log('Training SVM using reference data')

    svm_start = time.time()
    model, model_performance = optimize_svm(
        ref_u12_vector, 
        ref_u2_vector, 
        n_optimize=OPTIMIZE_N, 
        iterations=SVM_ITER,
        cv_jobs=CV_JOBS
    )
    svm_train_time = get_runtime(svm_start)
    if log is True:
        write_log(
            'SVM training finished in {}; F1 score on reference data: {}',
            svm_train_time, model_performance)

    # # TESTING: run trained SVM on ref data to print/plot ref scores
    # scored_refs = assign_svm_scores(
    #     scored_refs, model, scoring_regions, model_performance, add_type=False)

    # ref_vector = get_score_vector(
    #     raw_refs, score_names=scoring_region_labels)

    # if len(scoring_region_labels) > 1:
    #     scatter_plot(
    #         scored_refs,
    #         ref_vector,
    #         '{}_ref_scatterplot'.format(SPECIES_FULL),
    #         xlab=scoring_region_labels[0],
    #         ylab=scoring_region_labels[1]
    #     )

    # with open('{}.ref_introns.scores'.format(SPECIES), 'w') as ref_out:
    #     for i in scored_refs:
    #         dnts = '-'.join(i.dnts)
    #         ref_out.write('\t'.join(map(str, [i.name, i.type_id, dnts, i.svm_score, i.five_z_score, i.five_seq, i.bp_z_score, i.bp_seq, i.three_z_score, i.three_seq])))
    #         ref_out.write('\n')


    # run trained SVM on experimental data
    scored_introns = assign_svm_scores(
        scored_introns, model, scoring_regions, model_performance)

    flipped = get_flipped(scored_introns, model, score_scaler)
    if log is True:
        write_log(
            '{} putative U12 scores were not robust to boundary switching',
            len(flipped.keys()))

    finalized_introns = []
    u12_count = 0
    atac_count = 0
    demoted_swaps = []

    for i in demote(scored_introns, flipped):
        if '[d]' in i.dynamic_tag:
            demoted_swaps.append(i.demote_info)
        if i.svm_score > THRESHOLD:
            u12_count += 1
            if i.dnts == ('AT', 'AC'):
                atac_count += 1
        finalized_introns.append(i)
    
    return finalized_introns, model, u12_count, atac_count, demoted_swaps


def recursive_scoring(
    finalized_introns,
    refs,
    model,
    matrices,
    scoring_region_labels,
    raw_score_names,
    z_score_names):

    # use introns from first round to create new matrices
    # for second round
    write_log('Updating scoring matrices using empirical data')
    new_matrices = make_matrices(
        finalized_introns, 
        THRESHOLD, 
        FIVE_SCORE_COORDS[0], 
        THREE_SCORE_COORDS[0],
        min_seqs=5)

    new_matrices = add_pseudos(new_matrices, pseudo=PSEUDOCOUNT)
    if u12_count < 200:
        mod_type = 'Averaging'
        MATRICES = average_matrices(matrices, new_matrices)
    # replace old matrices with new ones
    else:
        mod_type = 'Replacing'
        matrices.update(new_matrices)
    write_log('{} matrices using empirically-derived data'.format(mod_type))

    # re-run previously-trained SVM models on introns scored with new 
    # matrices to filter out any introns whose scored changed dramatically 
    # before picking a species-specific reference set
    raw_introns = list(get_raw_scores(finalized_introns, matrices))
    raw_refs = list(get_raw_scores(refs, matrices))
    scale_vector = get_score_vector(
        raw_introns + raw_refs, score_names=raw_score_names)
    recursive_scaler = preprocessing.StandardScaler().fit(scale_vector)
    scored_introns = scale_scores(raw_introns, recursive_scaler)
    scored_introns = assign_svm_scores(
        scored_introns, model, scoring_region_labels)

    # filter reference introns to non-redundant set
    # before training SVM
    ref_u12_threshold = 95
    ref_u2_threshold = 50
    unambiguous = [
        copy.deepcopy(i) for i in scored_introns if 
        i.svm_score > ref_u12_threshold or i.svm_score < ref_u2_threshold]
    unambiguous_seqs = get_attributes(
        unambiguous, ['five_seq', 'bp_region_seq', 'three_seq'])
    recursive_refs = []
    unique_ref_seqs = set()
    for i, seqs in zip(unambiguous, unambiguous_seqs):
        if seqs not in unique_ref_seqs:
            unique_ref_seqs.add(seqs)
            recursive_refs.append(i)

    scale_vector = get_score_vector(
        raw_introns + recursive_refs, score_names=raw_score_names)

    recursive_scaler = preprocessing.StandardScaler().fit(scale_vector)

    scored_introns = scale_scores(raw_introns, recursive_scaler)
    scored_refs = scale_scores(recursive_refs, recursive_scaler)

    recursive_ref_u12s = [
        i for i in scored_refs if i.svm_score > ref_u12_threshold]
    recursive_ref_u2s = [
        i for i in scored_refs if i.type_id == 'u2']

    write_log(
        'Empirically-derived training data: {} U2, {} U12', 
        len(recursive_ref_u2s), len(recursive_ref_u12s))
    # ensure that the training set is sufficiently large
    # adding the reference introns with scores derived from the
    # initial matrices ensures that changes to the matrices do
    MIN_REF_U2 = 5000
    MIN_REF_U12 = 100

    ref_u2_delta = MIN_REF_U2 - len(recursive_ref_u2s)
    ref_u12_delta = MIN_REF_U12 - len(recursive_ref_u12s)
    random.seed(42)
    if ref_u2_delta > 0 or ref_u12_delta > 0:
        raw_default_refs = list(get_raw_scores(refs, matrices))
        scored_default_refs = scale_scores(raw_default_refs, recursive_scaler)
        working_default_refs = assign_svm_scores(
            scored_default_refs, model, scoring_region_labels)
        scored_ref_u2s = [
            i for i in working_default_refs if i.svm_score < ref_u2_threshold]
        scored_ref_u12s = [
            i for i in working_default_refs if i.svm_score > ref_u12_threshold]
    if ref_u2_delta > 0:
        write_log('[!] Adding {} reference U2s to meet minimum', ref_u2_delta)
        recursive_ref_u2s += random.sample(scored_ref_u2s, ref_u2_delta)
    if ref_u12_delta > 0:
        write_log('[!] Adding {} reference U12s to meet minimum', ref_u12_delta)
        recursive_ref_u12s += random.sample(scored_ref_u12s, ref_u12_delta)

    recursive_refs = recursive_ref_u2s + recursive_ref_u12s

    return recursive_refs, scored_introns, matrices

# SVM functions ##############################################################

# def train_models(
#     base_model,
#     u2_vectors,
#     u12_vectors,
#     iterations,
#     cv_parameters=None,
#     train_fraction=0.8,
#     seed=42):
#     np.random.seed(seed)
#     subset_size = len(u12_vectors)
#     u2s = np.array(u2_vectors)
#     u12s = np.array(u12_vectors)
#     u2_sets = np.random.choice(
#         len(u2s), size=(iterations, subset_size), replace=False)
#     u2_sets = u2s[u2_sets]
#     trained_models = []
#     performance_scores = []
#     input_labels = np.concatenate([np.zeros(subset_size), np.ones(len(u12s))])
#     for iter_n, u2_subset in enumerate(u2_sets, start=1):
#         prog_bar = progress_bar(iter_n, iterations)
#         percent_complete = round(iter_n / iterations * 100, 2)
#         sys.stdout.write(
#             '\rTraining progress: {} - {}% ({}/{}) complete'.format(
#                 prog_bar, percent_complete, iter_n, iterations))
#         if cv_parameters is not None:
#             model = GridSearchCV(
#             base_model,
#             **cv_parameters)
#         else:
#             model = clone(base_model)
#         input_feats = np.concatenate([u2_subset, u12s])
#         (train_scores, test_scores,
#          train_labels, test_labels) = train_test_split(
#             input_feats,
#             input_labels,
#             train_size=train_fraction)
#         model.fit(train_scores, train_labels)
#         predict_labels = model.predict(test_scores)
#         subset_f1 = f1_score(test_labels, predict_labels)
#         performance_scores.append(subset_f1)
#         trained_models.append(model)

#     print()  # leave space after last progress bar update
#     return trained_models, performance_scores


def rank_ones(cv_results, key):
    ranks = cv_results['rank_test_score']
    params = cv_results['params']
    rank_one_params = [e[0][key] for e in zip(params, ranks) if e[1] == 1]

    return rank_one_params


def index_of_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


# def rank1_param_avg(model):
#     """
#     Returns geometric mean of rank-1 parameter values across all models
#     """
#     params = defaultdict(list)
#     for p, v in model.best_params_.items():
#         best_params = rank_ones(model.cv_results_, p)
#         params[p].extend(best_params)
#     refined_params = {}
#     for p, v in params.items():
#         # use geometric mean to find a value in-between the bounds of the range
#         best_avg_param = pystats.gmean(v)
#         refined_params[p] = best_avg_param

#     return refined_params


def rank1_param_avg(models):
    """
    Returns geometric mean of rank-1 parameter values across all models
    """
#     params = defaultdict(Counter)
    params = defaultdict(list)
    for m in models:
        for p, v in m.best_params_.items():
            best_params = rank_ones(m.cv_results_, p)
            params[p].extend(best_params)
    refined_params = {}
    for p, v in params.items():
        # use geometric mean to find a value in-between the bounds of the range
        best_avg_param = pystats.gmean(v)
        refined_params[p] = best_avg_param

    return refined_params


# def iterative_training(
#     u12_vectors,
#     u2_vectors,
#     iterations=50,
#     seed=42,
#     scorer='f1',
#     n_optimize=3):
#     possible_iterations = math.floor(len(u2_vectors) / len(u12_vectors))
#     if possible_iterations < iterations:
#         iterations = possible_iterations
#         write_log(
#             '[!] Training iterations limited to {} '
#             'due to size of reference U2 set', iterations)
#     parameter_range_start = -6
#     parameter_range_stop = 6
#     log_intervals = np.logspace(
#         parameter_range_start,
#         parameter_range_stop,
#         num=abs(parameter_range_stop - parameter_range_start) + 1)

#     initial_parameters = {'C': log_intervals}

#     base_model = svm.SVC(
#         probability=True,
#         kernel='linear',
#         class_weight='balanced',
#         # kernel='poly',  ###!!!
#         # gamma='scale',  ###!!!
#         # degree=2,  ###!!!
#         cache_size=1000
#     )
#     cv_params = {
#         'cv': 5,
#         'iid': False,
#         'scoring': 'balanced_accuracy',
#         'n_jobs': 5,
#         'param_grid': initial_parameters
#     }
#     refined_params = initial_parameters
#     for search_round in range(1, n_optimize + 1):
#         round_seed = seed + search_round
#         print('Optimization round {}/{}'.format(search_round, n_optimize))
#         search_models, _ = train_models(
#             base_model,
#             u2_vectors,
#             u12_vectors,
#             cv_parameters=cv_params,
#             iterations=iterations,
#             seed=round_seed,
#         )
#         best_first_params = rank1_param_avg(search_models)
#         for p, v in best_first_params.items():
#             best_index = index_of_nearest(refined_params[p], v)
#             low_bound = refined_params[p][max(best_index - 1, 0)]
#             high_bound = refined_params[p][min(
#                 best_index + 1, len(refined_params[p]) - 1)]
#             p_range = np.geomspace(low_bound, high_bound, 100)
#             refined_params[p] = p_range
#         cv_params['param_grid'] = refined_params
        
#     min_hyper = min(cv_params['param_grid']['C'])
#     max_hyper = max(cv_params['param_grid']['C'])
#     write_log(
#         'Optimized range for hyperparameter \'C\': {}-{}', 
#         min_hyper, max_hyper
#     )
#     write_log(
#         'Building models with optimized hyperparameters ({} rounds)', 
#         iterations)
#     trained_models, performance_scores = train_models(
#         base_model,
#         u2_vectors,
#         u12_vectors,
#         cv_parameters=cv_params,
#         iterations=iterations,
#         seed=seed)

#     return trained_models, performance_scores

# iterations
def train_svm(
    base_model,
    u2_vector,
    u12_vector,
    iterations,
    cv_parameters=None,
    train_fraction=0.85,
    seed=42):
    u12s = np.array(u12_vector)
    u2s = np.array(u2_vector)
    np.random.seed(seed)
    if iterations:
        subset_size = len(u12_vector)
        u2_sets = np.random.choice(
            len(u2s), size=(iterations, subset_size), replace=False)
        u2_sets = u2s[u2_sets]
        zero_labels = np.zeros(subset_size)
    else:
        base_model.class_weight = 'balanced'
        u2_sets = [u2s]
        zero_labels = np.zeros(len(u2s))
    trained_models = []
    performance_scores = []
    input_labels = np.concatenate([zero_labels, np.ones(len(u12s))])
    for iter_n, u2_subset in enumerate(u2_sets, start=1):
        if iterations:
            prog_bar = progress_bar(iter_n, iterations)
            percent_complete = round(iter_n / iterations * 100, 2)
            sys.stdout.write(
                '\rTraining progress: {} - {}% ({}/{}) complete'.format(
                    prog_bar, percent_complete, iter_n, iterations))
        if cv_parameters is not None:
            model = GridSearchCV(
            base_model,
            **cv_parameters)
        else:
            model = clone(base_model)
        input_feats = np.concatenate([u2_subset, u12s])
        (train_scores, test_scores,
         train_labels, test_labels) = train_test_split(
            input_feats,
            input_labels,
            train_size=train_fraction)
        model.fit(train_scores, train_labels)
        predict_labels = model.predict(test_scores)
        # if len(set(predict_labels)) == 1:
            # subset_f1 = 0
        # else:
        subset_f1 = f1_score(test_labels, predict_labels)
        # other = 1 - log_loss(test_labels, predict_labels)
        # print('\r' + ','.join(map(str, [iteration_n, subset_f1, other])), end='')
        performance_scores.append(subset_f1)
        trained_models.append(model)
    
    if iterations:
        print()  # leave space after last progress bar update
    return trained_models, performance_scores

# # no iterations
# def train_svm(
#     base_model,
#     u2_vector,
#     u12_vector,
#     cv_parameters=None,
#     train_fraction=0.75):
#     u2s = np.array(u2_vector)
#     u12s = np.array(u12_vector)
#     input_labels = np.concatenate([np.zeros(len(u2s)), np.ones(len(u12s))])
#     if cv_parameters is not None:
#         model = GridSearchCV(
#         base_model,
#         **cv_parameters)
#     else:
#         model = clone(base_model)
#     input_feats = np.concatenate([u2s, u12s])
#     (train_scores, test_scores,
#         train_labels, test_labels) = train_test_split(
#         input_feats,
#         input_labels,
#         train_size=train_fraction)
#     model.fit(train_scores, train_labels)
#     predict_labels = model.predict(test_scores)
#     subset_f1 = f1_score(test_labels, predict_labels)

#     return model, subset_f1


# # no iterations
# def optimize_svm(
#     u12_vector,
#     u2_vector,
#     scorer='balanced_accuracy',
#     n_optimize=4,
#     range_subdivisions=50):
#     parameter_range_start = -6
#     parameter_range_stop = 6
#     log_intervals = np.logspace(
#         parameter_range_start,
#         parameter_range_stop,
#         num=abs(parameter_range_stop - parameter_range_start) + 1)

#     initial_parameters = {'C': log_intervals}

#     # make things run faster for testing
#     # u2_vector = random.sample(list(u2_vector), 5000)   ###!!!

#     base_model = svm.SVC(
#         probability=True,
#         kernel='linear',
#         class_weight='balanced',
#         # kernel='poly',  ###!!!
#         # gamma='scale',  ###!!!
#         # degree=2,  ###!!!
#         cache_size=1000
#     )
#     cv_params = {
#         'cv': 5,
#         'iid': False,
#         'scoring': scorer,
#         'n_jobs': 4,
#         'param_grid': initial_parameters
#     }
#     refined_params = initial_parameters
#     for search_round in range(1, n_optimize + 1):
#         sys.stdout.write(
#             '\rOptimization round {}/{}'.format(
#                 search_round, n_optimize))
#         search_model, performance = train_svm(
#             base_model,
#             u2_vector,
#             u12_vector,
#             cv_parameters=cv_params)
#         best_first_params = rank1_param_avg(search_model)
#         for p, v in best_first_params.items():
#             best_index = index_of_nearest(refined_params[p], v)
#             low_bound = refined_params[p][max(best_index - 1, 0)]
#             high_bound = refined_params[p][min(
#                 best_index + 1, len(refined_params[p]) - 1)]
#             p_range = np.geomspace(low_bound, high_bound, range_subdivisions)
#             refined_params[p] = p_range
#         cv_params['param_grid'] = refined_params
    
#     print()
#     min_hyper = min(cv_params['param_grid']['C'])
#     max_hyper = max(cv_params['param_grid']['C'])
#     write_log(
#         'Range for \'C\' after optimization: {}-{}',
#         min_hyper, max_hyper
#     )
#     avg_hyper = np.mean([min_hyper, max_hyper])
#     cv_params['param_grid']['C'] = avg_hyper
#     write_log(
#         'Set classifier value for \'C\': {}', 
#         avg_hyper
#     )
#     write_log('Training classifier with optimized hyperparameters')
#     base_model.C = avg_hyper
#     trained_model, performance = train_svm(
#         base_model,
#         u2_vector,
#         u12_vector)

#     return trained_model, performance

# iterations
def optimize_svm(
    u12_vector,
    u2_vector,
    scorer='f1',
    n_optimize=4,
    range_subdivisions=50,
    iterations=0,
    seed=42,
    cv_jobs=1):
    parameter_range_start = -6
    parameter_range_stop = 6
    log_intervals = np.logspace(
        parameter_range_start,
        parameter_range_stop,
        num=abs(parameter_range_stop - parameter_range_start) + 1)

    initial_parameters = {'C': log_intervals}

    # make things run faster for testing
    # u2_vector = random.sample(list(u2_vector), 5000)   ###!!!

    base_model = svm.SVC(
        probability=True,
        kernel='linear',
        # class_weight='balanced',
        # kernel='poly',  ###!!!
        # gamma='scale',  ###!!!
        # degree=2,  ###!!!
        cache_size=1000
    )
    cv_params = {
        'cv': 5,
        'iid': False,
        'scoring': scorer,
        'n_jobs': cv_jobs,
        'param_grid': initial_parameters
    }
    refined_params = initial_parameters
    for search_round in range(1, n_optimize + 1):
        round_seed = seed + search_round
        print('Starting optimization round {}/{}'.format(
                search_round, n_optimize))
        search_model, performance = train_svm(
            base_model,
            u2_vector,
            u12_vector,
            iterations=iterations,
            cv_parameters=cv_params,
            seed=round_seed)
        best_first_params = rank1_param_avg(search_model)
        for p, v in best_first_params.items():
            best_index = index_of_nearest(refined_params[p], v)
            low_bound = refined_params[p][max(best_index - 1, 0)]
            high_bound = refined_params[p][min(
                best_index + 1, len(refined_params[p]) - 1)]
            p_range = np.geomspace(low_bound, high_bound, range_subdivisions)
            refined_params[p] = p_range
        cv_params['param_grid'] = refined_params
    
    if iterations:
        print()
    min_hyper = min(cv_params['param_grid']['C'])
    max_hyper = max(cv_params['param_grid']['C'])
    write_log(
        'Range for \'C\' after {} rounds of optimization: {}-{}',
        n_optimize, min_hyper, max_hyper
    )
    avg_hyper = np.mean(cv_params['param_grid']['C'])
    cv_params['param_grid']['C'] = avg_hyper
    write_log(
        'Set classifier value for \'C\': {}', 
        avg_hyper
    )
    write_log('Training classifier with optimized hyperparameters')
    base_model.C = avg_hyper
    trained_model, performance = train_svm(
        base_model,
        u2_vector,
        u12_vector,
        iterations=iterations)
    
    if iterations is not None:
        performance = np.mean(performance)

    return trained_model, performance


# no iterations
# def svm_predict(score_vectors, model):
#     probability_ledger = defaultdict(list)
#     probabilities = model.predict_proba(score_vectors)
#     labels = model.predict(score_vectors)
#     for i, lp in enumerate(zip(probabilities, labels)):
#         probability_ledger[i] = lp

#     return probability_ledger

# iterations
def svm_predict(score_vectors, model_list):
    probability_ledger = defaultdict(list)
    for m in model_list:
        probabilities = m.predict_proba(score_vectors)
        labels = m.predict(score_vectors)
        for i, lp in enumerate(zip(probabilities, labels)):
            probability_ledger[i].append(lp)

    return probability_ledger


# iterations
def average_svm_score(probability_dict, weights=None):
    average_probs = defaultdict(dict)
    if weights is not None:
        weights = np.array(weights)
    for k, v in probability_dict.items():
        u2_probs = np.array([e[0][0] for e in v])
        u12_probs = np.array([e[0][1] for e in v])
        if weights is not None:
            # weight each average by the performance of its source classifier
            u2_probs = u2_probs * weights
            u12_probs = u12_probs * weights
        label_count = Counter([e[1] for e in v])
        # avg_u2_prob = np.mean(u2_probs)
        avg_u12_prob = np.mean(u12_probs)
        # u2_prob_stdev = np.std(u2_probs)
        # u12_prob_stdev = np.std(u12_probs)
        # u12_prob_sem = pystats.sem(u12_probs)
        average_probs[k] = {
            'u12_avg': avg_u12_prob,
            # 'u12_sem': u12_prob_sem,
            # 'u2_avg': avg_u2_prob,
            # 'u12_std': u12_prob_stdev,
            # 'u2_std': u2_prob_stdev,
            'labels': label_count
        }

    return average_probs


def get_attributes(objs, attr_names):
    if type(objs) is not list:
        objs = [objs]
    return [tuple([getattr(o, a) for a in attr_names]) for o in objs]


def get_score_vector(introns, score_names):
    vect = [[getattr(i, n) for n in score_names] for i in introns]
    
    return np.asarray(vect)


def u12_label_ratio(label_dict):
    u12 = label_dict[1]
    total = sum(label_dict.values())
    
    return u12 / total


# # no iterations
# def assign_svm_scores(introns, model, scoring_region_labels, weights=None):
#     intron_score_vector = get_score_vector(
#         introns, score_names=scoring_region_labels)

#     u12_probability_index = svm_predict(intron_score_vector, model)

#     id_map = {
#         0: 'u2',
#         1: 'u12'
#     }

#     for idx, intron in enumerate(introns):
#         score_info = u12_probability_index[idx]
#         u2_score, u12_score = score_info[0]
#         label = id_map[score_info[1]]
#         u12_score = u12_score * 100
#         intron.svm_score = u12_score
#         intron.relative_score = (u12_score - THRESHOLD) / THRESHOLD * 100
#         intron.type_id = label

#     return introns

# iterations
def assign_svm_scores(
    introns, 
    models, 
    scoring_region_labels, 
    weights=None,
    add_type=True):
    intron_score_vector = get_score_vector(
        introns, score_names=scoring_region_labels)

    u12_probability_index = svm_predict(intron_score_vector, models)

    avg_u12_probabilities = average_svm_score(
        u12_probability_index, weights)

    id_map = {
        0: 'u2',
        1: 'u12'
    }

    for idx, intron in enumerate(introns):
        intron.svm_score = avg_u12_probabilities[idx]['u12_avg'] * 100
        # intron.error = avg_u12_probabilities[idx]['u12_sem'] * 100
        # relative as percentage of threshold
        intron.relative_score = (intron.svm_score - THRESHOLD) / THRESHOLD * 100
        # relative as probability
        # intron.relative_score = -1 * (intron.svm_score - THRESHOLD)
        label_ratio = u12_label_ratio(avg_u12_probabilities[idx]['labels'])
        if add_type is True:
            if label_ratio > 0.5:
                type_id = 'u12'
            else:
                type_id = 'u2'
            intron.type_id = type_id
        intron.label_ratio = label_ratio

    return introns

# /SVM functions ######################################################

# Plotting functions #########################################################

def density_hexplot(
    scores, title, xlab=None, ylab=None, outfmt='png', fsize=14):
    # plt.rcParams['figure.figsize'] = [12, 10.2]
    plt.figure(figsize=(12, 10.2))
    hx = plt.hexbin(
        *scores.T, mincnt=1, cmap='inferno', bins='log', linewidths=0)
    cb = plt.colorbar(hx)
    cb.set_label('Bin density (log10(N))')
    title_with_n = '{} (n={})'.format(title, len(scores))
    if xlab:
        plt.xlabel(xlab, fontsize=fsize)
    if ylab:
        plt.ylabel(ylab, fontsize=fsize)
    plt.title(title_with_n, fontsize=fsize)
    plt.tight_layout()
    title = '_'.join(title.split())
    plt.savefig("{}.iic.{}".format(title, outfmt), format=outfmt, dpi=150)
    plt.close()


def scatter_plot(introns, scores, title, xlab, ylab, fsize=14, outfmt='png'):
    plt.figure(figsize=(12, 12))
    cluster_colors = []
    u2_count, u12_low, u12_med, u12_high = [0] * 4
    score_stdev = np.std([i.svm_score for i in introns])
    high_val = THRESHOLD
    med_val = THRESHOLD - score_stdev
    # low_val = THRESHOLD - (score_stdev * 2)
    for i in introns:
        itype = i.type_id
        p = i.svm_score
        if itype == 'u2':
            u2_count += 1
            color = 'xkcd:medium grey'
        elif p > high_val:
            u12_high += 1
            color = 'xkcd:green'
        elif med_val < p <= high_val:
            u12_med += 1
            color = 'xkcd:orange'
        elif p <= med_val:
            u12_low += 1
            color = 'xkcd:red'
        cluster_colors.append(color)

    legend_colors = [
        'xkcd:medium grey', 'xkcd:red', 'xkcd:orange', 'xkcd:green']
    # legend_labels = ['U2', 'U12<=68', '68<U12<=95', 'U12>95']
    round_vals = map(round, [high_val, med_val])
    legend_labels = [
        'U2', 
        'U12<={}'.format(int(med_val)), 
        '{}<U12<={}'.format(int(med_val), int(high_val)), 
        'U12>{}'.format(int(high_val))
    ]
    legend_counts = [u2_count, u12_low, u12_med, u12_high]
    legend_patches = []
    for label, count, color in zip(legend_labels, legend_counts, legend_colors):
        label = '{} ({})'.format(label, count)
        patch = mpatches.Patch(color=color, label=label)
        legend_patches.append(patch)
    plt.scatter(*scores[:,:2].T, s=42, c=cluster_colors, alpha=0.5)
    plt.legend(handles=legend_patches)
    plt.xlabel(xlab, fontsize=fsize)
    plt.ylabel(ylab, fontsize=fsize)
    plt.title(title, fontsize=fsize)
    plt.savefig('{}.iic.{}'.format(title, outfmt), format=outfmt, dpi=150)
    plt.close()


def histogram(data_list, title=None, grid=True, bins='auto', log=True):
    plt.figure(figsize=(13, 8))
    if log is True:
        plt.yscale('log')
    plt.hist(data_list, bins=bins)
    if grid:
        plt.grid(True, which="both", ls="--", alpha=0.7)
    if title is not None:
        plt.title(title, fontsize=14)
    plt.xlabel('U12 score', fontsize=14)
    plt.savefig('{}.iic.png'.format(title.replace(' ', '_')), dpi=150)

# /Plotting functions ########################################################

# /Functions #################################################################

# Main #######################################################################

# Get command-line arguments
parser = argparse.ArgumentParser(
    description='intronIC (intron Interrogator and Classifier) is a '
    'script which collects all of the annotated introns found in a '
    'genome/annotation file pair, and produces a variety of output '
    'files (*.iic) which describe the annotated introns and (optionally) '
    'their similarity to known U12 sequences.\n'
    'Without the \'-m\' flag, there MUST exist a matrix file in the '
    '\'intronIC_data\' subdirectory in the same parent directory '
    'as intronIC.py, with filename \'scoring_matrices.fasta.iic\'. '
    'In the same data directory, there must also be a pair of sequence '
    'files (see --format_info) with reference intron sequences named '
    '\'[u2, u12]_reference_set.introns.iic\'',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
req_parse_grp = parser.add_argument_group(
    title='required arguments (-g, -a | -q)')
req_parse_grp.add_argument(
    '-g',
    '--genome',
    help='Genome file in FASTA format (gzip compatible)')
req_parse_grp.add_argument(
    '-a',
    '--annotation',
    help='Annotation file in gff/gff3/gtf format (gzip compatible)')
req_parse_grp.add_argument(
    '-n',
    '--species_name',
    type=str,
    help='Binomial species name, used in output file and intron label '
    'formatting. It is recommended to include at least the first letter '
    'of the species, and the full genus name since intronIC (by default) '
    'abbreviates the provided name in its output '
    '(e.g. Homo_sapiens --> HomSap)',
    required=True)
req_parse_grp.add_argument(
    '-q',
    '--sequence_input',
    help=(
        'Provide intron sequences directly, rather than using a '
        'genome/annotation combination. Must follow the introns.iic '
        'format (see README for description)'))
parser.add_argument(
    '-e',
    '--use_exons',
    action='store_true',
    help='Use exon rather than CDS features '
    'to define introns')
parser.add_argument(
    '-s',
    '--sequences_only',
    action='store_true',
    help='Bypass the scoring system and simply report the intron '
    'sequences present in the annotations')
parser.add_argument(
    '-nc',
    '--allow_noncanonical',
    action='store_true',
    help='Do not omit introns with non-canonical '
    'splicing boundaries from scoring')
parser.add_argument(
    '-i',
    '--allow_multiple_isoforms',
    action='store_true',
    help='Include non-duplicate introns from isoforms other than '
    'the longest in the scored intron set')
parser.add_argument(
    '-v',
    '--allow_intron_overlap',
    action='store_true',
    help='Allow introns with boundaries that overlap other introns from '
    'higher-priority transcripts (longer coding length, etc.) to be '
    'included. This will include, for instance, introns with alternative '
    '5′/3′ boundaries')
parser.add_argument(
    '-m',
    '--custom_matrices',
    metavar='{matrix file}',
    help='One or more matrices to use in place of the defaults. '
    'Must follow the formatting described by the --format_info '
    'option')
parser.add_argument(
    '-r12',
    '--reference_u12s',
    metavar='{reference U12 intron sequences}',
    help='introns.iic file with custom reference introns to be used '
    'for setting U12 scoring expectation, including flanking regions')
parser.add_argument(
    '-r2',
    '--reference_u2s',
    metavar='{reference U2 intron sequences}',
    help='introns.iic file with custom reference introns to be used '
    'for setting U12 scoring expectation, including flanking regions')
parser.add_argument(
    '--no_plot',
    action='store_true',
    help='Do not output illustrations of intron scores/distributions'
    '(plotting requires matplotlib)')
parser.add_argument(
    '--format_info',
    action='store_true',
    help='Print information about the system '
    'files required by this script')
parser.add_argument(
    '-d',
    '--include_duplicates',
    action='store_true',
    help='Include introns with duplicate '
    'coordinates in the intron seqs file')
parser.add_argument(
    '-u',
    '--uninformative_naming',
    action='store_true',
    help='Use a simple naming scheme for introns instead of the '
    'verbose, metadata-laden default format'
)
parser.add_argument(
    '-na',
    '--no_abbreviation',
    action='store_true',
    help='Use the provided species name in full within the output files'
)
parser.add_argument(
    '-t',
    '--threshold',
    metavar='0-100',
    type=float,
    default=90,
    help='Threshold value of the SVM-calculated probability of being a U12 to '
    'determine output statistics')
parser.add_argument(
    '-ns',
    '--no_sequence_output',
    action='store_true',
    help='Do not create a file with the full intron sequences '
    'of all annotated introns')
parser.add_argument(
    '--five_score_coords',
    default=(-3, 9),
    metavar=('start', 'stop'),
    nargs=2,
    type=int,
    help=(
        'Coordinates describing the 5\' sequence to be scored, relative to '
        'the 5\' splice site (e.g. position 0 is the first base of the '
        'intron); half-closed interval [start, stop)'
    )
)
parser.add_argument(
    '--three_score_coords',
    default=(-5, 4),
    metavar=('start', 'stop'),
    nargs=2,
    type=int,
    help=(
        'Coordinates describing the 3\' sequence to be scored, relative to '
        'the 3\' splice site (e.g. position -1 is the last base of the '
        'intron); half-closed interval [start, stop)'
    )
)
parser.add_argument(
    '-bpc',
    '--branch_point_coords',
    default=(-45, -5),
    metavar=('start', 'stop'),
    nargs=2,
    type=int,
    help=(
        'Coordinates describing the region to search for branch point '
        'sequences, relative to the 3\' splice site (e.g. position -1 is the '
        'last base of the intron); half-closed interval [start, stop)'
    )
)
parser.add_argument(
    '-r',
    '--scoring_regions',
    help='Intron sequence regions to include in intron score calculations',
    default=('five', 'bp'),
    choices=('five', 'bp', 'three'),
    nargs='+'
)
parser.add_argument(
    '-b',
    '--abbreviate_filenames',
    action='store_true',
    help='Use abbreviated species name when creating '
    'output filenames')
parser.add_argument(
    '--recursive',
    action='store_true',
    help=(
        'Generate new scoring matrices and training data using '
        'confident U12s from the first scoring pass. This option may '
        'produce better results in species distantly related to the '
        'species upon which the training data/matrices are based, though '
        'beware accidental training on false positives. Recommended only '
        'in cases where clear separation between types is seen on the first '
        'pass')
)
parser.add_argument(
    '--subsample_n',
    default=0,
    type=int,
    help=(
        'Number of sub-samples to use to generate SVM classifiers; 0 uses the '
        'entire training set and should provide the best results; otherwise, '
        'higher values will better approximate the entire set at the expense '
        'of speed')
)
parser.add_argument(
    '--parallel_cv',
    default=1,
    type=int,
    help=(
        'Number of parallel processes to use during cross-validation; '
        'increasing this value will reduce runtime but may result in '
        'instability due to outstanding issues in scikit-learn')
)

format_info_message = ("""
# Matrix files ################################################################

/// Description ///

The matrix file describes the frequency of each nucleotide at each position
in scoring region for different intron types. SpliceRack has frequency
matrices available which describe region motifs for large collections of
introns from different species. The matrices used by default are from human.

/// Naming ///

The matrix file must be named 'scoring_matrices.fasta.iic', and placed in a
directory called 'intronIC_data' located in the main intronIC directory.

/// Format ///

Each matrix in the file consists of a FASTA header line, indicating
which type of intron the matrix represents ('u2', 'u12'), which region
it describes ('five', 'bp', 'three') and which intron subtype it applies to
('gtag', 'atac', etc). Optionally, this header may also include a special
keyword phrase 'start={integer}', where {integer} is the position of
the first matrix entry relative to the corresponding splice site. If the
5' matrix begins with three exonic positions upstream of the beginning of 
the intron, for example, then the header for that matrix would include
'start=-3'. If this keyword is omitted, the matrix is assumed to start at 
position 0, the first base of the intron. Because the branch point motif 
occurs at variable locations, branch point matrices do not use this 
convention.

The line after the header must be the whitespace-separated order of the bases 
as they appear in the subsequent lines of the matrix. The rest of the lines
under the header are tab-separated values (columns) representing the
frequency of each base at each position in the target sequence (rows).

Example formatting ('scoring_matrices.fasta.iic'):

>u12_gtag_five  start=-3
A   C   G   T
0.293478260869565	0.239130434782609	0.239130434782609	0.228260869565217
0.271739130434783	0.326086956521739	0.152173913043478	0.25
0.217391304347826	0.184782608695652	0.0543478260869565	0.543478260869565
0	0	1	0
0	0	0	1
[...]
>u12_gtag_bp
A   C   G   T
0.12	0.15	0.21	0.52
0.12	0.17	0.18	0.53
0.13	0.2 0.12	0.55
[...]

# Reference introns ###########################################################

/// Description ///

The reference intron set is a collection of introns of inferred types, 15000
U2s and 1084 U12s, which have been found to be conserved between human, mouse
and chicken and, in the case of the U12s, also including additional human U12s
from Tyler Alioto's U12DB database. Each intron will be scored against the
same matrices used for the experimental dataset to determine an optimal
U2/U12 scoring threshold.

/// Naming ///

The files needs to be named '[u2, u12]_reference_set.introns.iic[.gz]', and
be located in the intronIC_data directory, unless specified via the -r[2, 12]
command line argument.

/// Format ///

Each line in the file contains the
following columns:

1. Identifier
2. Score (or placeholder; unused)
3. Length (or placeholder; unused)
4. 5' exonic sequence (>= the length used for scoring, if any)
5. Intronic sequence
6. 3' exonic sequence (>= the length used for scoring, if any)
""")

# Args special cases
if len(sys.argv) == 1:
    parser.print_usage()
    sys.exit(
        '{}: error: must be run with either genome and annotation, '
        'or introns file'.format(os.path.basename(sys.argv[0])))

if '--format_info' in sys.argv:  # can't use argparse unless other args
    sys.exit(format_info_message)
args = parser.parse_args()
if args.format_info:
    sys.exit(format_info_message)

if args.sequence_input and (args.genome or args.annotation):
    parser.error(
        'Must specify either direct sequence input (via -q) or a '
        'genome/annotation combination'
    )

# External filenames
matrix_filename = "scoring_matrices.fasta.iic"
reference_u12s_filename = "u12_reference.introns.iic"
reference_u2s_filename = "u2_reference.introns.iic"
backup_u2_bps_filename = 'u2.conserved_bps_matrix.iic'

# Global constants ###########################################################

# Get script directory and external file paths
HOME = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_DIR = os.path.join(HOME, "intronIC_data")
RUN_DIR = os.getcwd()
MATRIX_FILE = os.path.join(DATA_DIR, matrix_filename)
U2_BPS_MATRIX_FILE = os.path.join(DATA_DIR, backup_u2_bps_filename)

# This is inelegant, but to do otherwise would require reorg of args
# and constants (defaulting the arg would require knowing the data dir
# and matrix filename, for example)
if args.reference_u12s:
    REFERENCE_U12_FILE = args.reference_u12s
else:
    ref_u12_fns = [
        os.path.join(DATA_DIR, r) for r in
        [reference_u12s_filename, reference_u12s_filename + '.gz']]
    REFERENCE_U12_FILE = next(r for r in ref_u12_fns if os.path.isfile(r))
if args.reference_u2s:
    REFERENCE_U2_FILE = args.reference_u2s
else:
    ref_u2_fns = [
        os.path.join(DATA_DIR, r) for r in
        [reference_u2s_filename, reference_u2s_filename + '.gz']]
    REFERENCE_U2_FILE = next(r for r in ref_u2_fns if os.path.isfile(r))

SVM_ITER = args.subsample_n
CV_JOBS = args.parallel_cv
RECURSIVE = args.recursive
# scoring region coordinates are relative to the 5' and 3' ends of the intron
FIVE_SCORE_COORDS = tuple(args.five_score_coords)
THREE_SCORE_COORDS = tuple(args.three_score_coords)
BP_REGION_COORDS = tuple(args.branch_point_coords)
OPTIMIZE_N = 5
INTRON_FLANK_SIZE = 50  # for sending to file
MIN_INTRON_LENGTH = 30
# number of nc splice sites to print to screen
NUM_NC_USER = 5
# number of nc splice sites to print to log
NUM_NC_LOG = 20
# value to add to matrix values to avoid div 0 errors
PSEUDOCOUNT = 0.001

# Stuff from args
GENOME = args.genome
ANNOTATION = args.annotation
SEQUENCE_INPUT = args.sequence_input
SPECIES_FULL = args.species_name  # used in log file header
SPECIES = args.species_name  # used to name files
if not args.no_abbreviation:
    SPCS = abbreviate(SPECIES)  # used within files
else:
    SPCS = SPECIES
if not args.use_exons:  # to define introns
    FEATURE = 'cds'
else:
    FEATURE = 'exon'
ALLOW_NONCANONICAL = args.allow_noncanonical
ALLOW_OVERLAP = args.allow_intron_overlap
if args.allow_multiple_isoforms:
    LONGEST_ONLY = False
else:
    LONGEST_ONLY = True
THRESHOLD = args.threshold
INCLUDE_DUPES = args.include_duplicates
ONLY_SEQS = args.sequences_only
NO_SEQS = args.no_sequence_output
CUSTOM_MATRICES = args.custom_matrices
SIMPLE_NAME = args.uninformative_naming
SCORING_REGIONS = args.scoring_regions

# Change file-naming variable if specified
if args.abbreviate_filenames:
    SPECIES = abbreviate(SPECIES)

# Output filenames
FN_SEQS = '../tmp/{}.introns.iic'.format(SPECIES)
FN_RANKINGS = '../tmp/{}.rankings.iic'.format(SPECIES)
# FN_TMPLST = '{}.list.iic.tmp'.format(SPECIES)
FN_LIST = '../tmp/{}.list.iic'.format(SPECIES)
FN_DUPE_MAP = '../tmp/{}.dupe_map.iic'.format(SPECIES)
FN_OVERLAP_MAP = '../tmp/{}.overlap.iic'.format(SPECIES)
FN_MATRICES = '../tmp/{}.matrices.iic'.format(SPECIES)
FN_META = '../tmp/{}.meta.iic'.format(SPECIES)
FN_SCORE = '../tmp/{}.score_info.iic'.format(SPECIES)
FN_SWAP = '../tmp/{}.demoted.iic'.format(SPECIES)
FN_ANNOT = '../tmp/{}.annotation.iic'.format(SPECIES)
FN_LOG = '../intronIC_logs/{}.log.iic'.format(SPECIES)

# Logging setup ##############################################################

# Default config used as a base
# Debug messasges go only to log file, not screen
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=FN_LOG,
                    format='[#] %(asctime)s | %(message)s',
                    datefmt='%Y.%m.%d-%H.%M.%S',
                    level=logging.DEBUG,
                    filemode='w')
# Add logging module for printing to stdout
screenlogger = logging.StreamHandler(stream=sys.stdout)
screenlogger.setLevel(logging.INFO)
# Set less-verbose print syntax for screen vs. log file
screenformatter = logging.Formatter('[#] %(message)s')
screenlogger.setFormatter(screenformatter)
# Add screen logger to root logger
logging.getLogger('').addHandler(screenlogger)

# /Logging setup #############################################################

# Determine whether we can plot figures, assuming it is asked for
WANT_PLOT = not args.no_plot
if WANT_PLOT and CAN_PLOT:
    PLOT = True
else:
    PLOT = False
    if WANT_PLOT:
        write_log('Matplotlib not detected; plotting disabled')

# Start processing files

full_path_args = [
    os.path.abspath(a) if os.path.exists(a) else a for a in sys.argv
]
write_log('Run command: {}', ' '.join(full_path_args))

# Load external scoring matrices
MATRICES = load_external_matrix(MATRIX_FILE)
if CUSTOM_MATRICES:
    custom_matrix = load_external_matrix(CUSTOM_MATRICES)
    custom_keys = ['-'.join(key) for key in custom_matrix.keys()]
    write_log('Custom matrices used: {}', ','.join(custom_keys))
    MATRICES.update(custom_matrix)

# Pseudocounts added to avoid division by 0 during scoring
MATRICES = add_pseudos(MATRICES, pseudo=PSEUDOCOUNT)

# Determine length of 5' sequence region from supplied matrices
# and use if shorter than >FIVE_LENGTH<
five_matrix_length = matrix_length(MATRICES[('u12', 'gtag', 'five')])
three_matrix_length = matrix_length(MATRICES[('u12', 'gtag', 'three')])

ARG_FIVE_LENGTH = abs(FIVE_SCORE_COORDS[0] - FIVE_SCORE_COORDS[1])
ARG_THREE_LENGTH = abs(THREE_SCORE_COORDS[0] - THREE_SCORE_COORDS[1])

FIVE_LENGTH = min(ARG_FIVE_LENGTH, five_matrix_length)
THREE_LENGTH = min(ARG_THREE_LENGTH, three_matrix_length)

# Notify user if desired length is incompatible with matrix
for l, arg, tag in zip(
    [FIVE_LENGTH, THREE_LENGTH],
    [ARG_FIVE_LENGTH, ARG_THREE_LENGTH],
    ['5\'', '3\'']
):
    if l < arg:
        write_log(
            '[!] Length of {} region limited to {} by scoring matrices',
            tag, l
        )

# check min intron length against total size
# of scoring matrices and adjust up if necessary
bp_key = next(k for k in MATRICES.keys() if 'u12' in k and 'bp' in k)
BP_MATRIX_LENGTH = matrix_length(MATRICES[bp_key])
# BP_MATRIX_LENGTH = matrix_length(MATRICES[('u12', 'gtag', 'bp', 'vA9')])

# TODO adjust the bp margin to account for change in interval openness
bp_margin = abs(BP_REGION_COORDS[1])

# calculate length required by matrices at either end of intron
intronic_five = len([e for e in range(*FIVE_SCORE_COORDS) if e >= 0])
intronic_three = len([e for e in range(*THREE_SCORE_COORDS) if e < 0])
intronic_three = max((BP_MATRIX_LENGTH + bp_margin), intronic_three)

# scoring_region_size = FIVE_LENGTH + BP_MATRIX_LENGTH + bp_margin
scoring_region_size = intronic_five + intronic_three

if MIN_INTRON_LENGTH < scoring_region_size:
    MIN_INTRON_LENGTH = scoring_region_size
    write_log(
        '[!] Minimum intron length set to {} due to size of scoring regions',
        MIN_INTRON_LENGTH
    )

# Capture start time of run to facilitate calculation of total runtime at end
START_TIME = time.time()

# /Global constants ##########################################################

# Initial data collection ####################################################

write_log('Starting run on {}', SPECIES_FULL)

# Collect reference introns, with flanking exonic region
# on either end
REF_U2S = get_reference_introns(
    ref_file=REFERENCE_U2_FILE,
    five_score_coords=FIVE_SCORE_COORDS,
    three_score_coords=THREE_SCORE_COORDS,
    bp_coords=BP_REGION_COORDS,
    type_id='u2'
)
REF_U12S = get_reference_introns(
    ref_file=REFERENCE_U12_FILE,
    five_score_coords=FIVE_SCORE_COORDS,
    three_score_coords=THREE_SCORE_COORDS,
    bp_coords=BP_REGION_COORDS,
    type_id='u12'
)

# remove any reference data with redundant scores
filter_attributes = ['five_seq', 'bp_region_seq', 'three_seq']

REF_U12S = list(unique_attributes(REF_U12S, filter_attributes))
REF_U2S = list(unique_attributes(REF_U2S, filter_attributes))

REFS = REF_U12S + REF_U2S

if not args.sequence_input:
    source_file = ANNOTATION
    # Check to make sure annotation file has desired feature type
    intron_defining_features = ('exon', 'cds')
    if not has_feature(ANNOTATION, FEATURE):
        write_log('[!] No {} features in annotation.', FEATURE)
        FEATURE = next(f for f in intron_defining_features if f != FEATURE)
        if not has_feature(ANNOTATION, FEATURE):
            write_log('[!] No {} features in annotation. Exiting now.', FEATURE)
            sys.exit(1)

    write_log('Using {} features to define introns', FEATURE)

    # Pull all annotation entries into a hierarchy of objects
    top_level_annots = annotation_hierarchy(
        ANNOTATION, FEATURE)

    # Make a dictionary index for all intron-defining objects in the
    # annotated heirarchy. This index will allow easy access to parent
    # objects when mucking about with the introns

    # Add transcripts here to allow longest-isoform tagging
    flat_annots = flatten(top_level_annots, feat_list=[FEATURE, 'transcript'])

    # Make intron object dictionary from the collected top-level objects,
    # including whatever duplicates might exist in the annotation
    all_introns, total_count = collect_introns(top_level_annots, FEATURE)

    if total_count == 0:
        write_log(
            '[!] ERROR: No intron sequences found. '
            'Annotation <-> genome ID mismatch, perhaps? Exiting now.')
        sys.exit(1)

else:
    source_file = args.sequence_input
    all_introns = introns_from_flatfile(
        source_file,
        FIVE_SCORE_COORDS,
        THREE_SCORE_COORDS,
        BP_REGION_COORDS,
        ALLOW_NONCANONICAL,
        hashgen=True,
        allow_overlap=False)
    all_introns = list(all_introns)
    final_introns = [i for i in all_introns if not i.omitted]
    total_count = len(final_introns)
    omit_count = len(all_introns) - total_count
    if omit_count > 0:
        write_log('{} introns omitted', omit_count)

write_log('{} introns found in {}', total_count, source_file)

if not SEQUENCE_INPUT:
    # Populate introns with sub-sequences and write full seqs to file
    # Also, tag introns with >omitted< if meet omission criteria
    # Omitted introns are written to same file as all other introns
    stats_omitted_counts = Counter()
    stats_corrected_count = 0
    stats_nc_types = Counter()  # omitted for non-canon bounds
    final_introns = []
    omitted_introns = []
    corrected_duplicates = []
    duplicate_count = 0
    # Only make seqs file if not argument preventing it
    if not NO_SEQS:
        seq_file = open(FN_SEQS, 'w')
    # Write the omitted introns to list file first
    if not ONLY_SEQS:
        meta_file = open(FN_META, 'w')
    list_file = open(FN_LIST, 'w')

    seqs_attribs = [
        'get_name',
        '-',  # score placeholder
        'length',
        'upstream_flank',
        'seq',
        'downstream_flank'
    ]
    list_attribs = [
        'region',
        'strand',
        'start',
        'stop'
    ]

    # build an index of duplicate introns keyed by (region, start, stop) of
    # the intron which superceded them to allow score propagation
    dupe_intron_index = defaultdict(set)
    overlap_index = defaultdict(set)
    # map unique_num attributes to final intron names for use in
    # dupe_map output
    intron_name_index = {}

    # these two dictionaries enable communication between get_sub_seqs()
    # and add_tags()
    intron_index = defaultdict(lambda: defaultdict(dict))
    longest_isoforms = {}

    # Iterate over generator with transient full sequences
    # Keep your wits about you here, given the number of flags at play
    for intron in get_sub_seqs(
        all_introns,
        flat_annots,
        GENOME,
        INTRON_FLANK_SIZE,
        FIVE_SCORE_COORDS,
        THREE_SCORE_COORDS,
        BP_REGION_COORDS):
        # Set omission status before generating headers
        intron.omit_check(
            MIN_INTRON_LENGTH, ALLOW_NONCANONICAL,
            ALLOW_OVERLAP, LONGEST_ONLY)
        # add all intron tags
        if not intron.omitted:
            intron, intron_index, longest_isoforms = add_tags(
                intron, intron_index, longest_isoforms,
                ALLOW_OVERLAP, LONGEST_ONLY)
        intron.omit_check(
            MIN_INTRON_LENGTH, ALLOW_NONCANONICAL,
            ALLOW_OVERLAP, LONGEST_ONLY)
        if not intron.omitted and intron.noncanonical:
            # tag non-canonical introns in label even if being scored
            intron.dynamic_tag.add('[n]')
        if not intron.omitted and not intron.longest_isoform:
            intron.dynamic_tag.add('[i]')
        if not ONLY_SEQS:
            if not intron.omitted and intron.duplicate is False:
                final_introns.append(intron)
                # compute hash of intron sequence to identify introns
                # across different annotations
            if intron.duplicate is False:
                    intron.md5 = md5(intron.seq.encode('utf-8')).digest()
        intron_name_index[intron.unique_num] = intron.get_name()
        if intron.duplicate is not False:
            duplicate_count += 1
            dupe_intron_index[intron.duplicate].add(intron.get_name())
            if intron.corrected:
                corrected_duplicates.append(intron)
            if not INCLUDE_DUPES:
                continue
        # keep a index of introns which are being omitted due to
        # overlap
        if intron.omitted == 'v':
            overlap_index[intron.overlap].add(intron.get_name())
        if intron.corrected:
            stats_corrected_count += 1
        # if intron is non-canonical (omitted or not), add to nc stats
        if intron.noncanonical and intron.omitted not in ('s', 'a'):
        # if intron.noncanonical and not intron.omitted:
            dnts = '-'.join(intron.dnts)
            stats_nc_types[dnts] += 1
        if ONLY_SEQS: # no scoring info in intron label
            intron.omitted = False
            list_string = write_format(
                intron, *list_attribs, 'get_name', fasta=False)
            list_file.write(list_string + '\n')
        # write omitted introns to list file (unscored)
        elif intron.omitted and intron.duplicate is False:
            stats_omitted_counts[intron.omitted] += 1
            if intron.omitted == 'n':
                dnts = '-'.join(intron.dnts)
                intron.dynamic_tag.add(dnts)
            list_string = write_format(
                intron, *list_attribs, 'get_label', fasta=False)
            list_file.write(list_string + '\n')
            if not ONLY_SEQS:
                meta_bits = [
                    'get_name',
                    '-',  # score placeholder for omitted introns
                    '-'.join(intron.dnts),
                    'phase',
                    # odds of two introns producing the same 128-bit hash in a 
                    # set of 1 trillion introns is ~1.44e-15
                    intron.md5.hex(),
                    'fractional_position',
                ]
                meta_string = write_format(
                    intron, *meta_bits, fasta=False)
                meta_file.write(meta_string + '\n')
        if not NO_SEQS:
            # Write all introns to file
            line = write_format(
                intron, *seqs_attribs, fasta=False)
            seq_file.write(line + '\n')

        intron.seq = None  # save space

    if not NO_SEQS:  # only close if opened
        seq_file.close()
    if not ONLY_SEQS:
        meta_file.close()
    if dupe_intron_index:
        with open(FN_DUPE_MAP, 'w') as dupe_map:
            for chosen, dupes in dupe_intron_index.items():
                name = intron_name_index[chosen]
                for d in dupes:
                    dupe_map.write('{}\t{}\n'.format(name, d))

    if not ALLOW_OVERLAP and overlap_index:
        with open(FN_OVERLAP_MAP, 'w') as ol_map:
            for chosen, overlapping in overlap_index.items():
                name = intron_name_index[chosen]
                for o in overlapping:
                    ol_map.write('{}\t{}\n'.format(name, o))
    list_file.close()

    write_log('{} introns with duplicate coordinates excluded', duplicate_count)

    # If they only want the intron sequences, exit after writing seq file
    if ONLY_SEQS:
        write_log(
            '{} intron sequences written to {}',
            (total_count - duplicate_count),
            FN_SEQS)
        run_time = get_runtime(START_TIME)
        write_log('Run finished in {}', run_time)
        sys.exit()

    if stats_omitted_counts:
        write_log(
            '{} introns omitted from scoring based on the following criteria:',
            sum(stats_omitted_counts.values()))
        write_log(
            '* short (<{} nt): {}', MIN_INTRON_LENGTH, stats_omitted_counts['s'],
            wrap_chars=None)
        write_log(
            '* ambiguous nucleotides in scoring regions: {}',
            stats_omitted_counts['a'],
            wrap_chars=None)
        write_log(
            '* non-canonical boundaries: {}', stats_omitted_counts['n'],
            wrap_chars=None)
        write_log(
            '* overlapping coordinates: {}', stats_omitted_counts['v'],
            wrap_chars=None)
        write_log(
            '* not in longest isoform: {}', stats_omitted_counts['i'],
            wrap_chars=None)

    if stats_nc_types:
        write_log(
            'Most common non-canonical splice sites:',
        )
        for stat in counter_format_top(stats_nc_types, NUM_NC_USER):
            print('[#] {}'.format(stat))
        for stat in counter_format_top(stats_nc_types, NUM_NC_LOG):
            write_log(stat, level='debug')

    if not final_introns:
        write_log(
            '[!] ERROR: No intron sequences found. '
            'Check for annotation/genome ID mismatch. Exiting now.')
        sys.exit(1)

    # Correct coordinates of defining features in annotation if any
    # misannotated AT-AC introns are found
    correct_annotation(
        final_introns + corrected_duplicates, flat_annots, ANNOTATION, RUN_DIR)

FINAL_INTRON_COUNT = len(final_introns)

write_log('{} introns included in scoring analysis', FINAL_INTRON_COUNT)

# /Initial data collection ###################################################

# Scoring ####################################################################
#TODO make cmdline arg to require bp be built from all introns (in case is
# u2 matrix in matrices file but don't want to use it)

BPX = None

# [ v1 ]
# physarum introns conserved as U12s in other species, using the
# pattern '.{4}TTTGA.{3}.{6,8}$'
# BPX = {
#     6: 1.371429,
#     7: 1.428571,
#     8: 1.2
# }
# [ v2 ]
# physarum introns conserved as U12s in other species plus ATACs
# from conserved regions using the pattern '.{4}TTTGA.{3}.{6,8}$'
# BPX = {
#     6: 1.354167, # 17/48
#     7: 1.4375, # 21/48
#     8: 1.208333  # 10 /48
# }
# [ v3 ]
# physarum introns conserved as U12s in others and BUSCO matches,
# plus ATACs in conserved regions
# BPX = {
#     6: 1.352941, # 54/153
#     7: 1.431373, # 66/153
#     8: 1.215686  # 33/153
# }

###!!! FIRST ROUND OF SCORING
MATRICES = add_u2_matrix(MATRICES, final_introns)

# Get the maximum and minimum score possible for each matrix, to be used
# to scale scores
MATRIX_BOUNDS = defaultdict(dict)
for k, m in MATRICES.items():
    category = k[:2]
    sub_category = k[-1]
    MATRIX_BOUNDS[category][sub_category] = get_score_bounds(m)

# working_introns = copy.deepcopy(final_introns)
# working_refs = copy.deepcopy(REFS)

# working_introns = final_introns
# working_refs = REFS

# # Get the raw log ratio scores of each scoring region in each intron
# raw_introns = get_raw_scores(working_introns, MATRICES, MATRIX_BOUNDS)

# # Same for the reference sequences
# raw_refs = get_raw_scores(working_refs, MATRICES, MATRIX_BOUNDS)
# write_log(
#     'Raw scores calculated for {} U2 and {} U12 reference introns',
#     len(REF_U2S), len(REF_U12S))

# get score stats for ref and experimental introns combined, then
# score introns based on those stats (to make z-scores comparable)

# raw_introns = list(raw_introns)
# raw_refs = list(raw_refs)
# all_intron_stats = get_raw_score_stats(raw_introns + raw_refs)

# # Convert raw scores to z scores base on statistics from all introns,
# # and scale raw scores into the range (0, 1)
# intron_score_stats, scored_introns = adjust_scores(
#     raw_introns, sample=False, stats=all_intron_stats)

# ref_score_stats, scored_refs = adjust_scores(
#     raw_refs, log=False, sample=False, stats=all_intron_stats)


# # Convert raw scores to z-scores based on individual set statistics
# raw_introns = list(raw_introns)
# ### temp
# intron_score_stats, scored_introns = adjust_scores(
#     raw_introns, sample=False)
# ## temp

# raw_refs = list(raw_refs)
# ### temp
# ref_score_stats, scored_refs = adjust_scores(
#     raw_refs, log=False, sample=False)
# ### temp

# train SVM using reference intron sets
# scoring_region_labels = SCORING_REGIONS

write_log(
    'Scoring introns using the following regions: {}',
    ', '.join(SCORING_REGIONS)
)

score_label_table = {
    'five': 'five_z_score',
    'bp': 'bp_z_score',
    'three': 'three_z_score'
}

# ensure that scoring regions are sorted in the correct order for plotting
scoring_region_labels = []
label_order = ['five', 'bp', 'three']
for label in label_order:
    if label in SCORING_REGIONS:
        scoring_region_labels.append(score_label_table[label])

#### temp
# scored_refs = list(scored_refs)

# scored_ref_u12s = [i for i in scored_refs if i.type_id == 'u12']
# scored_ref_u2s = [i for i in scored_refs if i.type_id == 'u2']

# ref_u12_vector = get_score_vector(
#     scored_ref_u12s, scoring_regions=scoring_region_labels)
# ref_u12_vector = np.unique(ref_u12_vector, axis=0)
# ref_u2_vector = get_score_vector(
#     scored_ref_u2s, scoring_regions=scoring_region_labels)
# ref_u2_vector = np.unique(ref_u2_vector, axis=0)
### temp




raw_score_names = ['five_raw_score', 'bp_raw_score', 'three_raw_score']
z_score_names = ['five_z_score', 'bp_z_score', 'three_z_score']

finalized_introns, model, u12_count, atac_count, demoted_swaps = apply_scores(
    REFS, final_introns, MATRICES, scoring_region_labels)

###!!! /FIRST ROUND OF SCORING

if RECURSIVE and u12_count > 5:
    r_refs, r_introns, r_matrices = recursive_scoring(
        finalized_introns, 
        REFS, 
        model, 
        MATRICES, 
        scoring_region_labels, 
        raw_score_names, 
        z_score_names)
    
    (
        finalized_introns, 
        model, 
        u12_count, 
        atac_count, 
        demoted_swaps
    ) = apply_scores(r_refs, r_introns, r_matrices, scoring_region_labels)

write_log(
    '{} putative AT-AC U12 introns found.', atac_count)
write_log(
    '{} putative U12 introns found with scores > {}%', u12_count, THRESHOLD)

# format and write output files

if demoted_swaps:
    with open(FN_SWAP, 'w') as swaps:
        for info in demoted_swaps:
            swaps.write(
                '{}\t{:.5f} ({:.5f}, {:.5f}, {:.5f}) --> '
                '{:.5f} ({:.5f}, {:.5f}, {:.5f})\n'.format(*info))

# Write matrices to file for reference,
# removing added pseudocounts first
print_matrices = {
    k: add_pseudos(v, pseudo=-PSEUDOCOUNT)
    for k, v in MATRICES.items()}
write_matrix_file(print_matrices, FN_MATRICES)

if not SEQUENCE_INPUT:
    int_list_file = open(FN_LIST, 'a')
    for i in finalized_introns:
        list_string = write_format(
            i, 'region', 'strand', 'start', 'stop', 'get_label', fasta=False)
        int_list_file.write(list_string + '\n')
    int_list_file.close()

ranking_file = open(FN_RANKINGS, 'w')
score_file = open(FN_SCORE, 'w')
meta_file = open(FN_META, 'a')

# devlin
devlin_dict = {}
#/ devlin

if SEQUENCE_INPUT:
    ref_format = True
else:
    ref_format=False

for index, intron in enumerate(sorted(
        finalized_introns, key=attrgetter('svm_score')), start=1):
    rank_string = ranking_format(intron, index=index, ref=ref_format)
    ranking_file.write(rank_string + '\n')

    # devlin
    devlin_bits = [
        intron.get_name(),  # intron label
        rank_string.split('|')[1],  # scoring regions with ellipses
        SPECIES_FULL,  # species name (as typed on the cmdline)
        intron.relative_score,
        intron.region,  # genomic header
        intron.start,
        intron.stop,
        intron.strand,
        intron.phase,
        intron.index,  # intron rank within transcript ("4/7", etc)
        '-'.join(intron.dnts),  # GT-AG, AT-AC, etc
        intron.upstream_flank,  # upstream exonic sequence
        intron.downstream_flank,  # downstream exonic sequence
        intron.bp_seq,  # scored branch point sequence
        intron.parent,  # transcript
        intron.grandparent  # gene (if present in the annotation)
    ]

    devlin_dict[intron.get_name().split(';')[0]] = devlin_bits
    #/ devlin

    score_bits = [
        intron.name,
        intron.relative_score,
        intron.svm_score,
        intron.five_seq,
        intron.five_raw_score,
        intron.five_z_score,
        intron.bp_seq,
        intron.bp_raw_score,
        intron.bp_z_score,
        intron.three_seq,
        intron.three_raw_score,
        intron.three_z_score,
    ]
    score_string = '\t'.join([str(e) for e in score_bits])
    score_file.write(score_string + '\n')

    meta_bits = [
        'get_name',
        str(round(intron.relative_score, 4)),
        '-'.join(intron.dnts),
        'phase',
        # odds of two introns producing the same 128-bit hash in a 
        # set of 1 trillion introns is ~1.44e-15
        intron.md5.hex(),
        'fractional_position',
        'type_id',
    ]
    meta_string = write_format(intron, *meta_bits, fasta=False)
    meta_file.write(meta_string + '\n')

score_file.close()
meta_file.close()
ranking_file.close()

if not NO_SEQS and not SEQUENCE_INPUT:
    write_log('Adding scores to intron sequences file')
    add_scores(finalized_introns, FN_SEQS)

if PLOT:
    # build lists of component scores for making plots
    score_vector = np.array(
        [(i.five_z_score, i.bp_z_score) for i in finalized_introns])

    write_log('Generating figures')
    # disable logging temporarily to avoid matplotlib writing
    # to root logging stream
    logging.disable(logging.INFO)
    # plot intron component scores with and without marked U12s
    density_hexplot(
        score_vector,
        '{}_hexplot'.format(SPECIES_FULL),
        xlab='5\' z-score',
        ylab='BPS z-score'
    )
    scatter_plot(
        finalized_introns,
        score_vector,
        '{}_scatterplot'.format(SPECIES_FULL),
        xlab='5\' z-score',
        ylab='BPS z-score'
    )
    summary_scores = [i.relative_score for i in finalized_introns]
    hist_title = '{} intron score distribution'.format(SPECIES_FULL)
    histogram(summary_scores, bins=100, title=hist_title)
    # re-enable logging
    logging.disable(logging.NOTSET)

run_time = get_runtime(START_TIME)

write_log('Run finished in {}', run_time)

logging.shutdown()

sys.exit(0)

# TODO report types of features that were skipped
# TODO modify functions to return dictionaries for extensibility?

# /Scoring ###################################################################
