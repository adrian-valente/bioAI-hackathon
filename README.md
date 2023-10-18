# bioAI-hackathon

Small demo project on exploiting protein and nucleotide language models to optimize coding DNA sequences for the production of a biosimilar product to a protein of interest (here demonstrated with the Humira antibody sequence). The pipeline involves the following steps: 
* Mutants were generated from the Humira sequence by masking safe locations predicted with Paragraph, and new amino-acids predicted by ESM2
* Biosimilar variants were selected by ABodyBuilder2
* Plasmids were generated through most-frequent-codon lookup table, then randomly mutated
* As a proxy for translation efficiency, a model to predict mRNA stability from DNA sequence was trained as a Lasso regression on top of the embeddings generated from one of Instadeep's nucleotide transformers.
* Variants with best predicted stability were selected.

Small 2-day project for the EF/Instadeep bio-AI [hackathon](https://lu.ma/jv79kyrq) in October 2023, done by @gamazeps, @glenzo, @OliverT1 and @tiffanybounmy.

