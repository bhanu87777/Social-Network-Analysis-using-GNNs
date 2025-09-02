<h1>Social Network Analysis using Graph Neural Network</h1>

<p>
This project implements social network analysis using Graph Neural Networks (GNNs). 
It leverages graph-based machine learning techniques to extract insights and patterns 
from social network data. The main objective is to explore relationships and influence 
patterns in networked data using GNN models. By representing social networks as graphs, 
the project captures both the structure and the attributes of entities, allowing for 
advanced analysis such as community detection, influence propagation, and prediction 
of unseen connections. 
</p>

<h2>How to Run</h2>

<pre><code>python main.py</code></pre>

<p>
By default, the program will load the dataset, build the graph, train the GNN model, 
and generate visualizations as outputs.
</p>

<h2>What this Project Does</h2>

<p>
This project uses the <b>Cora dataset</b> from Kaggle, which contains around 2,700 
scientific papers categorized into 7 different classes. Each paper is described by 
a set of attributes, and the citation links form the graph structure. 
</p>

<p>
The project finds similarities between papers and clusters them into seven distinct 
groups based on their research area. Using Graph Neural Networks, it captures both 
the content (features of papers) and the citation links (relationships) to classify 
and group them effectively.
</p>

<ul>
  <li>Model complex relationships between entities (papers) in the citation network</li>
  <li>Identify communities and clusters of papers across 7 classes</li>
  <li>Predict missing connections and paper classifications</li>
  <li>Visualize network structures and patterns</li>
</ul>

<p><i>Unstructured data is processed and transformed into graph representations for training.</i></p>

<img src="Image1" alt="Description of Image" width="800" />

<p>
The implementation includes data preprocessing, graph construction, 
GNN model training, and result visualization components.
</p>

<p><i>Clustering shown in PCA component:</i></p>

<img src="Image2" alt="Description of Image" width="800" />

<h2>Final Result</h2>

<p>The analysis produces:</p>

<ul>
  <li>Network visualizations showing community structures</li>
  <li>Node classification and link prediction results</li>
  <li>Influence metrics for important nodes in the network</li>
  <li>Clustering of 2,700 papers into 7 groups based on their research category</li>
  <li>Comparative analysis of different GNN architectures</li>
</ul>

<h2>Conclusion</h2>

<p>
This project demonstrates the power of Graph Neural Networks for social network analysis, 
providing insights into relationship patterns and influence propagation that would be 
difficult to obtain with traditional methods. By applying GNNs to the Cora dataset, 
it successfully classifies and clusters scientific papers into 7 meaningful groups, 
highlighting the ability of GNNs to learn from both structure and attributes. 
The implementation serves as a foundation for more advanced network analysis applications. 
</p>
