import pandas as pd
import numpy as np
import calendar
import re
import seaborn as sns
import networkx as nx
import netwulf as nw
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import holoviews as hv
from holoviews import opts
import hvplot.networkx as hvnx
from functools import reduce
from sensible_raw.loaders import loader
from scipy.stats import pearsonr
#from bokeh.models import HoverTool

import inspect, re

def varname(p):
    """Takes the name of the variable DID NOT END UP WORKING AS INTENDED"""
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
        return m.group(1)

def get_datatype_period(datatype, period, sort=False):
    """Get data for a datatype and list of periods."""
    df = pd.DataFrame()
    for month in period:
        df = pd.concat([df, loader.load_data(datatype, month, as_dataframe=True)])
    return df

def netwulf_vis(G): #Keeps on posting the wrong network for some reason. 
    network, config = nw.visualize(G, port=8967, plot_in_cell_below=False)
    fig, ax = nw.draw_netwulf(network)
    #plt.colorbar(degreeColor)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    plt.show()

def create_nw_csv(data, name):
    """Creates a csv of all the connection in the raw data and stores the weights for the links
       as sent and recieved variables. Also filters away everyone who is not a part of the study
       by removing numbers above 852 (typing errors) and under 0 (people from outside the study)"""
    if 'address' not in data.columns:
        data.rename(columns={'number': 'address'}, inplace=True)
    D = data.sort_values(["user", "address"])
    D = D[(D.address > -1) & (D.user > -1) & (D.user < 852) & (D.address < 852)]
    D = D.reset_index(drop=True) #testing on only users in the study
    #calls_2 = calls
    users = []
    address = []
    sent= []
    recieved = []
    recipro = []
    freq_sum = []
    created_utc = []
    #temp_df = calls_2[calls_2["user"] == 0]

    #users=sms["user"].tolist()
    addresses=D["address"].tolist()
    participants = users + addresses
    participants = set(participants)


    for user in D["user"].unique():
        temp_df = D[D["user"] == user] #Looking at one user
        temp_df = temp_df.groupby(['address', 'type']).size().reset_index(name='Freq') # grouping by type to find frequency
        numbers = temp_df["address"].unique() #defining all the unique numbers the user has interracted with
        for i in range(len(numbers)):
            users.append(user)
            address.append(numbers[i])
            temp = temp_df[temp_df["address"] == numbers[i]]
            lst = [0,0] #making sure, that every list contains 2 elements
            for j in range(len(temp.address.to_list())):
                #Try statement, because i found a "5" element in the "type" column. Probably a typpo. 
                try:
                    lst[temp.iloc[j]["type"] - 1] = temp.iloc[j]["Freq"]
                except IndexError:
                    continue
            sent.append(lst[0])
            recieved.append(lst[1])
            freq_sum.append(sum(lst))
            if lst[0] > 3 and lst[1] > 3: #defining that people are friends from outgoing and incoming phone calls
                #Appending 1 if reciprocated
                recipro.append(1)
            elif lst[0] > lst[1] and lst[0] > 3:
                #appending 2 if not reciprocated
                recipro.append(2)
            else:
                recipro.append(0)
    print(len(users))
    print(len(address))
    #print(len(freqlst))
    print(len(recipro))
    print(len(freq_sum))


    data = {"user": users,
            "address": address,
            "sent": sent,
            "recieved": recieved,
            "connected": recipro,
            "freq_sum": freq_sum} # Defining the new data

    new_df = pd.DataFrame(data)
    #new_df = new_df.drop(['Unnamed: 0'], axis=1)
    new_df.to_csv("{}_network.csv".format(name))    
    
    
    
def initialize_network(data, source, target, nx=nx):
    """Initializes the networkx network from a dataframe. applies different attributes such as
       Pagerank and the sent values. Also updates the size according to the degree of the node
       and color from the pagerank algorithm"""
    nw = nx.from_pandas_edgelist(data, source=source, target=target, edge_attr=None, create_using = nx.DiGraph())
    nw.remove_edges_from(list(nx.selfloop_edges(nw))) # remove self loops
    weights = dict(zip(list(nw.edges), data.sent.values.tolist()))
    nx.set_edge_attributes(nw, weights, "weight")
    pr = nx.pagerank(nw, alpha=0.65) # compute pagerank
    sort_orders = sorted(pr.items(), key=lambda x: x[0], reverse=False)
    pr_sort = [val for (node, val) in sort_orders] # compute its degree
    #prepare
    degreeColor =[]
    valueToPlot = np.log10(pr_sort)
    # we rescale the colours to be in the RGB format (0 to 255 for three colours)
    valueToPlotRescaled = 255*(valueToPlot - np.min(valueToPlot))/np.max(valueToPlot)
    for i in valueToPlot:
        color = '#%02x%02x%02x' % (int(i), 0, 50) # here we use (X,0,50) as RGB with X beign the log(degree) for eahc node
        degreeColor.append(color)
    # zip it up into a dictionary and set it as node attribute
    dictionaryColor = dict(zip(list(nw.nodes), degreeColor))
    nx.set_node_attributes(nw, dictionaryColor, "group")
    d = dict(nw.degree)
    nx.set_node_attributes(nw, d, "size")
    pr = nx.pagerank(nw, alpha=0.65)
    pr.update((x, y*100) for x, y in pr.items())
    nx.set_node_attributes(nw, pr, "pagerank")
    return nw

def visualise_network_spring(G, iterations=1000):
    """Visualises a networkx network by using the spring layout"""
    hv.extension('bokeh')
    print('Graph construction complete.')
    pos = nx.spring_layout(G, k=0.8, iterations=iterations)
    return hvnx.draw(G, pos, node_size=hv.dim('size')*30, node_color=hv.dim('pagerank')*20, alpha=0.65).opts(width=1250, height=1250, edge_line_width=0.1)

def get_stats(G):
    """Get the different basic stats from a Networkx network"""
    degrees = [d for i,d in G.degree()]
    in_degrees = [d for i,d in G.in_degree()]
    out_degrees = [d for i,d in G.out_degree()]
    L = G.number_of_edges()
    L_avg = np.mean(G.number_of_edges())
    N = G.number_of_nodes()
    p = (L_avg)/(N*(N-1)/2)
    # From equation 3.3 the average degree is calculated
    k_avg = p*(N-1) 
    print("Amount of nodes in network:", nx.number_of_nodes(G))
    print("Amount of edges in network:", nx.number_of_edges(G))
    print(f"The probability that two nodes are connected: \t\t p = {np.round(p*100, 6)} %")
    print(f"The average degree of the nodes are: \t\t\t <k> = {np.round(k_avg, 4)}")
    print(f"Standard deviation of degrees of the network:\t {np.std(degrees):.4f}")
    print("Max degree:", np.max(degrees))
    print("Min degree:", np.min(degrees))
    print("Max In degree:", np.max(in_degrees))
    print("Min In degree:", np.min(in_degrees))
    print("Max Out degree:", np.max(out_degrees))
    print("Min Out degree:", np.min(out_degrees))
    print("Dangling Nodes:",len([node for node in G.nodes if G.out_degree(node) == 0]),"\n")
    
def degree_dist(G, name):
    """Plots the the degree distribution of a Networkx network"""
    degrees = [d for i,d in G.degree()]
    bins = np.logspace(min(degrees),np.log10(max(degrees)),20)
    hist, edges = np.histogram(degrees, bins, density=True)
    x = (edges[1:] + edges[:-1])/2.
    x, y = zip(*[(i,j) for (i,j) in zip(x, hist) if j > 0])

    fig, ax = plt.subplots(figsize = (14,6))
    ax.plot(x, y, label = "The {} network".format(name), marker = ".", color="red")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("P(k)", fontsize=15)
    ax.set_xlabel("Degree, k", fontsize=15)
    ax.set_title("Distributions of degrees in the {} network".format(name), fontsize=20)
    ax.legend()

    plt.show()
    
def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq

def degree_hist(G):
    """Plots a degree histogram of a networkx network"""
    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)
    degrees = range(len(in_degree_freq))
    plt.figure(figsize=(12, 8)) 
    plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree') 
    plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
    plt.title('Frequency Histogram of degrees',fontsize=16)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()

def rankings_df(txtfile):
    """Takes in a textfile name of rankings and converts the ranks to a pandas dataframe"""
    my_file = open("ranking/"+txtfile+".scc.el", "r")
    content1 = my_file.read()
    my_file.close()
    my_file = open("ranking/"+txtfile+".scc", "r")
    content2 = my_file.read()
    my_file.close()
    res = content1.replace('\n', ' ').split()
    numbers = [ int(x) for x in res ]
    res2 = content2.replace('\n', ' ').split()
    numbers2 = [ int(x) for x in res2 ]
    a = np.concatenate((np.array(numbers), np.array(numbers2)), axis=None)
    a = np.unique(a)
    
    my_file = open("ranking/"+txtfile+".avrank", "r")
    content = my_file.read()
    #print(content)
    content_list = content.replace('\n', ' ').split()
    my_file.close()
    b = list(map(float, content_list))
    c = []
    for i in range(len(b)):
        if i in a:
            c.append(b[i])
        else:
            c.append(-1)

    norm = [float(i)/max(c) for i in c]
    data = {"nodes":a,
            "ranks":np.array(c)}
    df = pd.DataFrame({'Participant': range(len(c)), 'Rank': np.array(c), 'Rank_norm':norm})
    return df
    
def ranking_distribution(df):
    """Plots the Ranks and the normalised tanks of a dataframe of ranks"""
    sns.histplot(df[df['Rank']>0].Rank, bins=int(50), kde=True).set_title('Distribution of Ranks')
    plt.show()


def calculate_pvalues(df):
    """Calculates the pvalues of a dataframe"""
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues    
    
def calc_pvals(df):
    """Calculates the P-value for each correlation in a pandas dataframe with asterisk notation of significance"""
    rho = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
    return rho.round(2).astype(str) + p

def network_ranking_df(G, rankings):
    """Creates a correlation heatmap of a network together with grades."""
    df1 = pd.DataFrame.from_dict(dict(G.degree), orient='index', columns=["degree"])
    pr = nx.pagerank(G, alpha=0.6)
    df2 = pd.DataFrame.from_dict(pr, orient='index', columns=["Pagerank"])
    GPA = pd.read_csv("GPA.csv")
    GPA = GPA.rename(columns={"user": "Participant"})
    gender = pd.read_csv("user_metadata.csv")
    gender = gender.rename(columns={"user": "Participant"})
    in_degrees = [d for i,d in G.in_degree()]
    out_degrees = [d for i,d in G.out_degree()]
    df3 = pd.DataFrame(np.array(in_degrees)/np.array(out_degrees), columns=["in_out"])
    #ranks = rankings.drop(['Unnamed: 0'], axis=1)
    
    gender = pd.read_csv("user_metadata.csv")
    gender = gender.rename(columns={"user": "Participant"})
    gender = gender.dropna()
    gender['gender'] = gender['gender'].map(lambda x: re.sub(r'[^-A-Z-^]','', x))
    from sklearn import preprocessing
    lbl = preprocessing.LabelEncoder()
    gender['gender'] = lbl.fit_transform(gender['gender'].astype(str))
    
    vectors = pd.read_csv('vectors.csv')
    users = np.zeros(853)
    for i in range(len(vectors)):
        users[vectors.vector[i]] = 1
    
    data = {'Participant': range(853),
            'vector': users}
    
    vector = pd.DataFrame(data)
    
    
    df1['Participant'] = df1.index
    df2['Participant'] = df1.index
    df3['Participant'] = df1.index
    
    data_frames = [df1, df3, GPA, gender, vector, df2, rankings]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Participant'], how='outer'), data_frames)
    df_merged = df_merged.drop(['Rank_norm','class'], axis=1)
    df_merged = df_merged.dropna()
    return df_merged

def correlation_heatmap(G, rankings):
    """Creates a correlation heatmap of a network together with grades."""
    df_merged = network_ranking_df(G, rankings).drop(['in_out','Participant'], axis=1)
    print("Top 5 highest ranked:")
    print(df_merged.nlargest(5, 'Rank'))
    print("Top 5 lowest ranked:")
    print(df_merged.nsmallest(5, 'Rank'))
    print(calc_pvals(df_merged))
    sns.heatmap(df_merged.corr(), vmin=-1, vmax=1, annot=True)
    
def create_edgelist(df, name, thres=0):
    """ Creates an edgelist of a dataframe with a wished threshold for how many messages are
        are considered to be a friendship"""
    lst = []
    print("Generating {} Edgelist with threshold of {}...".format(varname(df),thres))
    for i in tqdm(range(len(df))):
        if df.sent[i] > thres:
            string = "{} {}".format(df.user[i],df.address[i])
        elif df.recieved[i] > thres:
            string = "{} {}".format(df.address[i],df.user[i])
        else:
            continue
        lst.append(string)
    myset = set(lst)
    print(len(myset))
    newlist = list(myset)
    if thres == 0:
        textfile = open("ranking/{}_edgelist.txt".format(name), "w")
    else:
        textfile = open("ranking/{}_edgelist_thres{}.txt".format(name,thres), "w")
    print("Writing unique Edgelist to txt file...")
    for element in tqdm(newlist):
        textfile.write(element + "\n")
    textfile.close()
    return print("Done!")

def probability_of_friendship(rankings_edgelist):
    """Caluclates the emperical probability of an edge between two nodes based on the rankings
       of a network"""
    my_file = open("ranking/{}.twoway".format(rankings_edgelist), "r")
    content = my_file.read()
    content_list = content.split()
    my_file.close()
    b_twoway = list(map(float, content_list))
    
    my_file = open("ranking/{}.oneway".format(rankings_edgelist), "r")
    content = my_file.read()
    content_list = content.split()
    my_file.close()
    b_oneway = list(map(float, content_list))
    
    chunks_oneway = [b_oneway[x:x+3] for x in range(0, len(b_oneway), 3)]
    chunks_twoway = [b_twoway[x:x+3] for x in range(0, len(b_twoway), 3)]
    
    Data_oneway = {'z_value':np.array(chunks_oneway).T[0],
        'expected number of edges':np.array(chunks_oneway).T[1],
        'possible number of edges':np.array(chunks_oneway).T[2],
        'probability of friendship':np.array(chunks_oneway).T[1]/np.array(chunks_oneway).T[2]}

    df_oneway = pd.DataFrame(Data_oneway)
    
    Data_twoway = {'z_value':np.array(chunks_twoway).T[0],
        'expected number of edges':np.array(chunks_twoway).T[1],
        'possible number of edges':np.array(chunks_twoway).T[2],
        'probability of friendship':np.array(chunks_twoway).T[1]/np.array(chunks_twoway).T[2]}

    df_twoway = pd.DataFrame(Data_twoway)
    fig, ax =plt.subplots(1,2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    sns.lineplot(data=df_twoway, x="z_value", y="probability of friendship", ax=ax[0])
    sns.lineplot(data=df_oneway, x="z_value", y="probability of friendship", ax=ax[1])
    fig.show()
    
def create_time_csv(data, name):
    """Creates multiple edg """
    if 'address' not in data.columns:
        data.rename(columns={'number': 'address'}, inplace=True)
    D = data.sort_values(["user", "address"])
    D = D[(D.address > -1) & (D.user > -1) & (D.user < 852) & (D.address < 852)]
    D = D.reset_index(drop=True) 
    D["created_date"] = pd.to_datetime(D.timestamp.div(1000), unit='s').dt.date
    D = D.drop(['timestamp', 'status'], axis=1)
    
    D = D.set_index(D['created_date'])
    D = D.sort_index()
    Intervals = ['2012-01-01:2012-06-01','2012-06-01:2012-12-31','2013-01-01:2013-06-01',
                 '2013-06-01:2013-12-31','2014-01-01:2014-06-01','2014-06-01:2014-12-31',
                 '2015-01-01:2015-06-01','2015-06-01:2015-12-31','2016-01-01:2016-06-01',
                 '2016-06-01:2016-12-31']
    
    for i in range(len(Intervals)):
        x = Intervals[i].split(':')
        split_df = D[pd.to_datetime(x[0]).date():pd.to_datetime(x[1]).date()]
        #print(split_df)
        create_nw_csv(split_df,'time/{}_{}'.format(name,i))
        df = pd.read_csv('time/{}_{}_network.csv'.format(name,i))
        create_edgelist(df,'time/{}_{}'.format(name,i))
        
def visualise_network_ranking(G, rankings, size_scale=1, iterations=1000):
    """Visualises a networkx network by using the spring layout"""
    hv.extension('bokeh')
    print('Graph construction complete.')
    #ranks = rankings.drop(['Participant','Rank_norm'],axis=1).T.to_dict('list')
    #ranks = dict(zip(sorted(list(G.nodes), key=lambda x: float(x)), rankings.Rank.values.tolist()))
    data = {'Participant':range(852)}
    parti = pd.DataFrame(data)
    data_frames = [parti, rankings]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Participant'], how='outer'), data_frames)
    df_merged = df_merged.drop(['Rank_norm'], axis=1)
    ranks = df_merged.to_dict()['Rank']
    
    nx.set_node_attributes(G, ranks, "ranking")
    pos = nx.spring_layout(G, k=0.8, iterations=iterations)
    return hvnx.draw(G, pos, node_size=hv.dim('size')*size_scale, node_color=hv.dim('ranking')*20, alpha=0.65, edge_size=hv.dim('linkalpha')).opts(width=1250, height=1250, edge_line_width=0.1)

def visualise_in_out_networks(df, rankings):
    """Splits the network into a sent and recieved network and visualises it"""
    sent_df = df[df['sent'] >= df['recieved']]
    recieved_df = df[df['sent'] <= df['recieved']]
    sent_nw = initialize_network(sent_df, 'user', 'address')
    recieved_nw = initialize_network(recieved_df, 'user', 'address')
    idx_list_sent = sorted(list(sent_nw.nodes), key=lambda x: float(x))
    idx_list_recieved = sorted(list(sent_nw.nodes), key=lambda x: float(x))
    df1 = df.groupby(by=["user"]).sum()
    size_sent = dict(zip(sorted(list(sent_nw.nodes), key=lambda x: float(x)),df1.reset_index()[df1.reset_index()['user'].isin(idx_list_sent)].sent.values.tolist()))
    size_recieved = dict(zip(sorted(list(recieved_nw.nodes), key=lambda x: float(x)), df1.reset_index()[df1.reset_index()['user'].isin(idx_list_recieved)].recieved.values.tolist()))
    link_alpha_sent = dict(zip(list(sent_nw.edges), sent_df.sent.values.tolist()))
    link_alpha_recieved = dict(zip(list(recieved_nw.edges), recieved_df.recieved.values.tolist()))
    nx.set_node_attributes(sent_nw, size_sent, "size")
    nx.set_node_attributes(recieved_nw, size_recieved, "size")
    nx.set_edge_attributes(sent_nw, link_alpha_sent, "linkalpha")
    nx.set_edge_attributes(recieved_nw, link_alpha_recieved, "linkalpha")
    s = visualise_network_ranking(sent_nw, rankings)
    r = visualise_network_ranking(recieved_nw, rankings)
    return (s + r).cols(2)