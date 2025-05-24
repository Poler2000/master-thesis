# Application of data sketches in the analysis of large graphs

## Abstract 

Data Sketches are a powerful tool for analyzing large datasets, including graphs.
In this thesis, we define the graph streaming model and review existing solutions
in this field. We focus on methods that create embeddings of graph nodes. In particular, we discuss the NodeSketch algorithm, which uses samples generated from
exponential distribution to recursively create sketches based on the k-neighborhood
of a node. We then propose a way to augment NodeSketch with the different sketching scheme, based on ExpSketch algorithm, which allows more operations to be
performed on data sketches. We call resulting algorithm EdgeSketch. We provide
a theoretical analysis of its complexity and perform extensive experiments to evaluate it in practice. Results from node reconstruction task show that EdgeSketch
consistently outperforms NodeSketch in terms of precision.

**Keywords**: data sketches, graph analysis, data streams


Master Thesis created under the supervision of dr inż. Jakub Lemiesz.

Faculty of Fundamental Problems of Technology

Wrocław University of Science and Technology
