        # Why size of ActorActor is 1 more than node list's length
        # Why arrLabel/initial probability is zero for query nodes
        # Why accumulate the query nodes rows in ActorActor matrix

        -----------------------------------------------------------
        TO DO
        -----------------------------------------------------------

        Give uniform probability to all nodes (even the query nodes)
        
        Change the RWR prob value after each full run as the diameter of the
        graph changes (use the relation from the paper (1-c)^d relation)

        Merge the query nodes (by sum) & by maxm.

        From the modelpickle with word similarities how to get the 
        probabilities from them. Another problem is right now < 0 is clamped to 
        0. 

        Ex 1.
        gensim package -> word2vec -> get cosine sim (b.w 0 and 1) b.w. the vecs and use it as prob

        --------------------------------------------------------------
        Supervised part
        --------------------------------------------------------------

        Use multiple measurements other than word2vec e.g. morphological score

        Use supervised learning to rank these similarity measures 
        
        1 possible way of learning:
        	Run full RWRs with each measure seperately and pick a word from the DCS pick solution
        	and weight the measures by 1/rank in each of the rank list(produced by the RWRs with above measures)

        	Use linear reg. or logisitic reg. with L-BFGS (sklearn has it) for regularization

       	
        --------------------------------------------------------------
        Improvements
        --------------------------------------------------------------
        1. Cache the files, lengths, lang. model, probs. etc.
        2. How to handle words not in corpus / cbow model
            * Should be probability be zero


