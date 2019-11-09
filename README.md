This package takes a whole-genome sequence file and an annotation file (a GTF or GFF3 file in most cases) and uses them to create an SQLite database of intron annotation information. This package uniquely annotates intron class for every intron annotated in the annotation file, so it can be used to identify all U12-dependent introns in any genome with annotated introns. It also provides some wrapper functions for the sorts of SQL queries that would be useful for querying the database. If you use this, cite https://doi.org/10.1101/620658.

Argument descriptions for create_db.py:

Argument Name        | Description
-------------------: | --------------------------------------------------------------------------------------------------------------
-g --genome          | E.g. GRCh38 or hg38 for the latest version of the human genome.
-t --tax_name        | E.g. Homo_sapiens; be sure to use snake_case or CamelCase.
-c --common_name     | E.g. human or rhesus_macaque; be sure to use snake_case or CamelCase.
-a --annotation      | Path to a gtf or gff3 file containing annotation information for the genome of interest. This pipeline was built using ones from Ensembl, but should work on any annotation files.
-s --sequence        | Path to a whole-genome FASTA file.
-gs --gene_symbols   | If your genome has annotation information on Ensembl, you can pull gene names out of Biomart if you know which Biomart division (default, GRCh37, plants, metazoa, fungi) the genome is in by providing the name of the biomart division as the value to this argument (Googling the genome name and the word "ensembl" is the fastest way to find out which division it's in. If your genome is not annotated in Biomart and you still want to be able to search for introns using their gene, you can provide a path to a tab-delimited file with gene IDs (as annotated in the annotation file you provide) in the first column and gene symbols in the second column. If you don't want to be able to search for introns by gene names and/or don't want to deal with the Biomart thing, leave this blank.

Argument descriptions for search_functions.py:

Argument Name                | Description
---------------------------: | -------------------------------------------------------------------------------------------------------------
-u12 --U12_search            | Single string that will be tokenized and used as input to a full-text search of the U12-type intron annotation information. Generally works best if you give it gene names/symbols, terminal dinucleotides, or Ensembl gene or transcript ids.
-g --genomes                 | One or more genome assembly names (e.g. GRCh38 for human; whatever you used when building the database), separated by commas.
-gid --gene_id               | Ensembl gene ID; separate multiple IDs with commas.
-tid --transcript_id         | Ensembl transcript ID; separate multiple IDs with commas.
-gs --gene_symbol            | Gene symbol or name (as annotated by Ensembl); separate multiple names with commas.
-c --intron_class            | U12-type or U2-type.
-p --phase                   | 0 if intron is between two codons, 1 if between first and second nucleotides of one codon, 2 if between second and third nucleotides.
-l --length                  | Intron length in nt/bp; will match exactly.
-lmin --min_length           | Min intron length in nt/bp.
-lmax --max_length           | Max intron length in nt/bp.
-r --rank                    | Intron rank in transcript; e.g. the first intron in a transcript is rank 1, the second is rank 2, etc. Matches exactly.
-rmin --min_rank             | Min intron rank.
-rmax --max_rank             | Max intron rank.
-chr --chromosome            | Chromosome containing intron.
-s --strand                  | Strand containing intron (usually + or -, but maybe 1 or -1 depending on where your gtf came from).'
-b --start                   | Start coordinate of intron in genome (b stands for beginning). Matches exactly.
-bmin --min_start            | Min intron start coordinate.
-bmax --max_start            | Max intron start coordinate.
-e --start                   | Stop coordinate of intron in genome (e stands for end). Matches exactly.
-emin --min_stop             | Min intron stop coordinate.
-emax --max_stop             | Max intron stop coordinate.
-td --terminal_dinucleotides | First two and last two nucleotides of the intron; most are GT-AG, a large minority of U12-types are AT-AC, and others are generally quite rare. If specifying multiple sets of terminal dinucleotides, separate them with commas. Do not neglect the hyphen in the middle.
