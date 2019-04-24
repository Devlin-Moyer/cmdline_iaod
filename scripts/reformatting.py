# reformatting.py
# takes output of intronIC, get_gene_symbols, and ortholog_clusters and gets it
# into a well-organized SQL database

import re
import sys
import sqlite3
import peewee
from playhouse.sqlite_ext import FTS5Model
from itertools import islice

# delete and re-create appropriate tables
def update_tables(genome, cur):
    # replace whatever data exists for this genome assembly
    cur.execute(f'DROP TABLE IF EXISTS {genome}')
    cur.execute(f'CREATE VIRTUAL TABLE {genome} USING fts5(short_seq \
UNINDEXED, tax_name, com_name, genome, score UNINDEXED, intron_class UNINDEXED,\
intron_id, chromosome UNINDEXED, start UNINDEXED, stop UNINDEXED, length \
UNINDEXED, strand UNINDEXED, intron_rank UNINDEXED, phase UNINDEXED, tds \
UNINDEXED, up_seq UNINDEXED, branch_seq UNINDEXED, down_seq UNINDEXED, \
full_seq UNINDEXED, gene_symbol, gene_id UNINDEXED, trans_id UNINDEXED)')
    # make U12s tables if the database is empty
    cur.execute('CREATE VIRTUAL TABLE IF NOT EXISTS U12s USING fts5(short_seq \
UNINDEXED, tax_name, com_name, genome, score UNINDEXED, intron_class \
UNINDEXED, intron_id, chromosome UNINDEXED, start UNINDEXED, stop UNINDEXED, \
length UNINDEXED, strand UNINDEXED, intron_rank UNINDEXED, phase UNINDEXED, \
tds, up_seq UNINDEXED, branch_seq UNINDEXED, down_seq UNINDEXED, full_seq \
UNINDEXED, gene_symbol, gene_id, trans_id)')
    # remove all the U12 introns from this genome assembly before adding more
    cur.execute('DELETE FROM U12s WHERE genome = ?', (genome,))
    cur.execute('COMMIT')

def insert_annotation(cur, genome, tax_name, com_name, symbols, batch):
    # makes all of these insertions happen within a single transaction- v fast
    cur.execute('BEGIN TRANSACTION')
    for line in batch:
        fields = line.rstrip('\n').split('\t')

        # these fields need to be re-arranged; name each element to reorder
        (intron_id, short_seq, bad_genome, score, chromosome, start, stop,
        strand, phase, rank, tds, up_seq, down_seq, branch_seq, trans_id,
        gene_id, full_seq) = fields

        # we've made bad_genome because we'll get <genome>_exon or <genome>_cds
        # from fields, and we just want the genome assembly name in the website

        # need to get intron_class using score and length using full_seq
        intron_class = ''
        if float(score) >= 0:
            intron_class = 'U12-Dependent'
        else:
            intron_class = 'U2-Dependent'

        length = len(full_seq)

        # need to find the gene symbol using symbols and gene_id; we won't find
        # a gene symbol for every gene_id
        try:
            gene_symbol = symbols[gene_id]
        except KeyError:
            gene_symbol = '.'

        # insert all the information
        cur.execute(f'INSERT INTO {genome} VALUES (?,?,?,?,?,?,?,?,?,\
?,?,?,?,?,?,?,?,?,?,?,?,?)', (short_seq, tax_name, com_name, genome, score,
intron_class, intron_id, chromosome, start, stop, length, strand, rank, phase,
tds, up_seq, branch_seq, down_seq, full_seq, gene_symbol, gene_id, trans_id))
        if intron_class == "U12-Dependent":
            cur.execute(f'INSERT INTO U12s VALUES (?,?,?,?,?,?,?,?,?,\
?,?,?,?,?,?,?,?,?,?,?,?,?)', (short_seq, tax_name, com_name, genome, score,
intron_class, intron_id, chromosome, start, stop, length, strand, rank, phase,
tds, up_seq, branch_seq, down_seq, full_seq, gene_symbol,gene_id, trans_id))
    cur.execute('COMMIT')

# get command-line arguments
if len(sys.argv) != 4:
   sys.exit('Specify the genome assembly name, the taxonomic name (delimited \
by underscores), and the common name (also delimited by underscores)')

genome = sys.argv[1]
bad_tax_name = sys.argv[2]
bad_com_name = sys.argv[3]

# remove underscores from tax_name and com_name
tax_name = re.sub('_', ' ', bad_tax_name)
com_name = re.sub('_', ' ', bad_com_name)

# get connection to database
conn = sqlite3.connect('../introns.db')
cur = conn.cursor()

# remove existing data for this genome assembly and/or create new tables for it
update_tables(genome, cur)

# make dictionary of gene IDs and gene symbols; batch input file because it may
# get really long with some genomes
print(f'Reading gene symbols.')
symbols = {}
with open(f'info/{genome}_gene_symbols.tsv', 'r') as in_file:
    for batch in iter(lambda: tuple(islice(in_file, 10000)), ()):
        for line in batch:
            id_symbol_pair = line.rstrip('\n').split('\t')
            symbols[id_symbol_pair[0]] = id_symbol_pair[1]

# get all of the annotation information from intronIC
with open(f'info/{genome}_info.tsv', 'r') as in_file:
    i = 0
    for batch in iter(lambda: tuple(islice(in_file, 10000)), ()):
        i += 1
        print(f'On batch {i}.')
        insert_annotation(
            cur, genome, tax_name, com_name, symbols, batch)
