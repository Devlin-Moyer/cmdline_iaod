# search_functions.py
# these functions need to be able to take command-line arguments with flags
# and execute them as SQL queries against the database generated from intronIC
# data, then create a BED file for all of the introns that come up in response
# to that search query

import argparse
import re
import datetime

# take a single string and use it to run a full-text query against the table
# that only contains information about U12-type introns
def U12_list(cur, args):
    user_query = args.q
    if user_query is not None:
        # the U12-search always returns the same columns and does a full-text
        # search, so it's very easy to handle these queries
        cur.execute(f'SELECT intron_id, genome, tax_name, com_name, \
gene_symbol, gene_id, trans_id, tds FROM U12s WHERE U12s MATCH {user_query}')
        hits = cur.fetchall()

        # convert list of tuples into a string that is a BED file
        bed_output = '\n'.join(['\t'.join(row) for row in hits])
        return(bed_output)

    else: # if they submitted a blank query, tell them to try a bit harder
        sys.exit('No search string detected; try "AT-AC Arabidopsis" to see \
all U12-type introns starting with AT and ending with AC in the Arabidopsis \
genome, assuming you built the database with the default IAOD genomes.')

# take a bunch of strings specified using command-line flags and construct sql
# queries and execute them against all specified genomes
def default_list(cur, args):
    # go through all the command-line options and add all appropriate wheres
    where_clauses = 'WHERE'
    if args.gene_id is not None:
        where_clauses += f' gene_id IN ({args.gene_id}) AND'
    if args.transcript_id is not None:
        where_clauses += f' trans_id IN ({args.transcript_id}) AND'
    if args.terminal_dinucleotides is not None:
        where_clauses += f' tds IN ({args.terminal_dinucleotides}) AND'
    if args.gene_symbol is not None:
        symbols = re.sub(',', ' OR ', args.gene_symbol)
        where_clauses += f" gene_symbol MATCH '{symbols}' AND"
    if args.intron_class is not None:
        where_clauses += f' intron_class = {args.intron_class} AND'
    if args.phase is not None:
        where_clauses += f' phase = {args.phase} AND'
    if args.length is not None:
        where_clauses += f' length = {args.length} AND'
    if args.rank is not None:
        where_clauses += f' intron_rank = {args.rank} AND'
    if args.chromosome is not None:
        where_clauses += f' chromosome = {args.chromsome} AND'
    if args.strand is not None:
        where_clauses += f' strand = {args.strand} AND'
    if args.start is not None:
        where_clauses += f' start = {args.start} AND'
    if args.stop is not None:
        where_clauses += f' stop = {args.stop} AND'
    if args.min_length is not None:
        where_clauses += f' length >= {args.min_length} AND'
    if args.min_rank is not None:
        where_clauses += f' rank >= {args.min_rank} AND'
    if args.min_start is not None:
        where_clauses += f' start >= {args.min_start} AND'
    if args.min_stop is not None:
        where_clauses += f' stop >= {args.min_stop} AND'
    if args.max_length is not None:
        where_clauses += f' length <= {args.max_length} AND'
    if args.max_rank is not None:
        where_clauses += f' rank <= {args.max_rank} AND'
    if args.max_start is not None:
        where_clauses += f' start <= {args.start} AND'
    if args.max_stop is not None:
        where_clauses += f' stop <= {args.stop} AND'

    # now that we've gone through all possible WHERE conditions, we need to
    # drop the last AND
    where_clauses = where_clauses.rstrip(' AND')

    # fill up an empty list of hits from all of the genomes
    hits = []
    genome_list = args.genomes.split(',')
    for genome in genome_list:
        cur.execute(f'SELECT chromosome, start, stop, intron_id, score, \
strand FROM {genome} {where_clauses}')
        some_hits = hits = cur.fetchall() # list of tuples where each tuple is
        # a row and each element of the tuple is a column
        hits.extend(some_hits)

    # convert list of tuples into a string that is a BED file
    bed_output = '\n'.join(['\t'.join(row) for row in hits])
    return(bed_output)

# start by making our command-line arguments

parser = argparse.ArgumentParser(description='Execute queries against the \
database of intron annotation and orthology information created with intronIC')

# boolean flags to determine what kind of search we're doing
parser.add_argument('-u12', '--U12_search', action='store_true',
    help='Execute a full-text search query against only the U12-type introns \
in the database. Accepts only a single string following -q. Outputs BED file \
of all U12-type introns returned as hits.')
parser.add_argument('-d', '--default_search', action='store_true',
    help='Execute a query against the entire database of intron annotation \
information. Outputs BED file of all introns matching criteria.')

# actual arguments for search queries
parser.add_argument('-q', '--U12_search',
    help='Single string that will be tokenized and used as input to a \
full-text search of the U12-type intron annotation information. Generally \
works best if you give it gene names/symbols, terminal dinucleotides, or \
Ensembl gene or transcript ids.')
parser.add_argument('-g', '--genomes',
    help='One or more genome assembly names (e.g. GRCh38 for human; whatever \
you used when building the database), separated by commas.')
parser.add_argument('-gid', '--gene_id',
    help='Ensembl gene ID; separate multiple IDs with commas.')
parser.add_argument('-tid', '--transcript_id',
    help='Ensembl transcript ID; separate multiple IDs with commas.')
parser.add_argument('-gs', '--gene_symbol', help='Gene symbol or name \
(as annotated by Ensembl); separate multiple names with commas.')
parser.add_argument('-c', '--intron_class',
    help='U12-type or U2-type.')
parser.add_argument('-p', '--phase', type='int',
    help='0 if intron is between two codons, 1 if between first and second \
nucleotides of one codon, 2 if between second and third')
parser.add_argument('-l', '--length', type='int',
    help='Intron length in nt/bp; will match exactly.')
parser.add_argument('-lmin', '--min_length', type='int',
    help='Min intron length in nt/bp.')
parser.add_argument('-lmax', '--max_length', type='int',
    help='Max intron length in nt/bp.')
parser.add_argument('-r', '--rank', type='int',
    help='Intron rank in transcript; e.g. the first intron in a transcript is \
    rank 1, the second is rank 2, etc. Matches exactly.')
parser.add_argument('-rmin', '--min_rank', type = 'int',
    help='Min intron rank.')
parser.add_argument('-rmax', '--max_rank', type='int',
    help='Max intron rank.')
parser.add_argument('-chr', '--chromosome',
    help='Chromosome containing intron.')
parser.add_argument('-s', '--strand', choices=['+', '-'],
    help='Strand containing intron (usually + or -, but maybe 1 or -1 \
depending on where your gtf came from).')
parser.add_argument('-b', '--start', type='int',
    help='Start coordinate of intron in genome (b stands for beginning). \
Matches exactly.')
parser.add_argument('-bmin', '--min_start', type='int',
    help='Min intron start coordinate.')
parser.add_argument('-bmax', '--max_start', type='int',
    help='Max intron start coordinate.')
parser.add_argument('-e', '--stop', type='int',
    help='Stop coordinate of intron in genome (e stands for end). \
Matches exactly.')
parser.add_argument('-emin', '--min_stop', type='int',
    help='Min intron stop coordinate.')
parser.add_argument('-emax', '--max_stop', type='int',
    help='Max intron stop coordinate.')
parser.add_argument('-td', '--terminal_dinucleotides',
    help='First two and last two nucleotides of the intron; most are GT-AG, a \
large minority of U12-types are AT-AC, and others are generally quite rare. If\
 specifying multiple sets of terminal dinucleotides, separate them with \
 commas. Do not neglect the hyphen in the middle.')

# get connection to database and then determine if we're executing a simple U12
# only search or a more complex main search

conn = sqlite3.connect('introns.db')
cur = conn.cursor()

# will hold BED-formatted output string
output = ''
args = parser.parse_args()
if args.u12 is True:
    output = U12_list(cur, args)
elif args.d is True:
    output = default_list(cur, args)

# make an output filename with the current timestamp to guarantee uniqueness
output_file = 'introns_' + re.sub(' ', '_', str(datetime.datetime.now()))

with open(output_file + '.bed', 'w') as out_file:
    out_file.write(output)
