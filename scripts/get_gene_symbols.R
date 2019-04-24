# get_gene_symbols.R
# uses biomaRt to find gene symbols for a list of ensembl gene IDs

library("biomaRt")

# turn a binomial name into a biomart dataset name
get_dataset <- function(tax_name, type) {
  # this is the only difference between dataset names in different marts
  thing <- ifelse(
    type == "default" | type == "GRCh37",
    "gene_ensembl",
    "eg_gene"
  )
  names <- strsplit(tax_name, "_")[[1]]
  dataset <- paste(
    paste(tolower(substr(names[1], 0, 1)), names[2], sep = ""),
    thing,
    sep = "_"
  )
  return(dataset)
}

# use awk to get a list of gene IDs from intronIC output
get_gene_ids <- function(genome) {
  gene_ids <- system2(
    "awk",
    c("'{print $15}'", paste("info/", genome, "_info.tsv", sep = "")),
    stdout = TRUE
  )
  return(unique(gene_ids))
}

# get gene symbols for a list of gene ids
query_biomaRt <- function(tax_name, gene_ids, type) {
  # make host and mart names assuming a variant biomaRt
  host_name <- paste(type, "ensembl.org", sep = ".")
  mart_name <- paste(type, "mart", sep = "_")
  # if the type is default, convert host and mart names to defaults
  host_name <- ifelse(
    type == "default" | type == "GRCh37",
    "ensembl.org",
    host_name
  )
  mart_name <- ifelse(
    type == "default" | type == "GRCh37",
    "ENSEMBL_MART_ENSEMBL",
    mart_name
  )
  gene_symbols <- getBM(
    attributes = c("ensembl_gene_id", "external_gene_name"),
    filters = "ensembl_gene_id",
    values = gene_ids,
    mart = useMart(
      mart_name,
      host = host_name,
      dataset = get_dataset(tax_name, type)
    )
  )
  return(gene_symbols)
}

# get command-line arguments
args = commandArgs(trailingOnly = TRUE)
type <- args[1]
genome <- args[2]
tax_name <- args[3]

if(!(type %in% c("default", "plants", "fungi", "metazoa", "GRCh37"))) {
  stop("biomaRt type can only be default, GRCh37, plants, fungi, or metazoa.")
}

cat("Finding gene names in", genome, "using", type, "biomart.\n")

write.table(query_biomaRt(tax_name, get_gene_ids(genome), type),
  # make helpful filename for output file
  file = paste("info/", genome, "_gene_symbols.tsv", sep = ""),
  # options to make it easy to parse this file with Python later
  quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE
)
