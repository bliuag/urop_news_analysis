# -*- coding: utf-8 -*-

import os
import pickle
import operator
import numpy as np

from global_var import *

def ad_hoc_process_doc(loc_file_doc):
    print "process doc from", loc_file_doc
    import datetime
    
    idx_date = 0
    idx_sec = 2
    idx_title = 3
    idx_summ = 4
    
    date_min = datetime.datetime.strptime("2016-12-12 12:12:12", "%Y-%m-%d %H:%M:%S")
    date_max = datetime.datetime.strptime("1880-12-12 12:12:12", "%Y-%m-%d %H:%M:%S")
    cnt = 0
    
    f = open(loc_file_doc)
    f.readline()
    for line in f:
        try:
            items = line.strip().split("\t")
            date = datetime.datetime.strptime(items[idx_date], "%Y-%m-%d %H:%M:%S")
            
            if date<date_min:
                date_min = date
            if date>date_max:
                date_max = date
                
            cnt+=1
        except BaseException as e:
            print e
    
    print date_min
    print date_max
    print cnt

def process_research(loc_file_research, flag_load_from_tmp_file=False):
    loc_tmp_file = os.path.join(g_loc_tmp, "research.tmp.pkl")
    if flag_load_from_tmp_file and os.path.isfile(loc_tmp_file):
        print "load research map from", loc_tmp_file
        with open(loc_tmp_file) as f:
            map_sid_date_r = pickle.load(f)
        return map_sid_date_r
        
    print "process research data from", loc_file_research
    import datetime
    
    idx_sid = 0
    idx_date = 1
    idx_research = 5
    label_pos = [u"跑赢大市", u"优于大市", u"超强大市", u"积极申购", u"建议申购", u"买入", u"增持", u"推荐", u"强烈推荐", u"强烈买入"]
    label_neg = [u"谨慎申购", u"落后大市", u"减持", u"回避", u"卖出"]
    label_neu = [u"同步大市", u"大市同步", u"持有", u"中性", u"观望", u"谨慎看好", u"审慎推荐"]
    
    map_sid_date_r = {}
    f = open(loc_file_research)
    f.readline()
    for line in f:
        items = line.strip().split("\t")
        sid = items[idx_sid].split("_")[0]
        date = datetime.datetime.strptime(items[idx_date], "%Y-%m-%d %H:%M:%S")
        items_research = items[idx_research].split("^")
        r_txt = items_research[1].decode("utf-8")
        agent = items_research[4].decode("utf-8")
        
        label = 0
        if r_txt in label_pos:
            label = 1
        elif r_txt in label_neg:
            label = -1
        elif r_txt in label_neu:
            label = 0
        else:
            if len(r_txt)>1:
                print r_txt
            continue
            
        if sid in map_sid_date_r:
            map_sid_date_r[sid].append( (date, label, agent) )
        else:
            map_sid_date_r[sid] = [(date, label, agent)]
    f.close()
    with open(loc_tmp_file, "w") as f:
        pickle.dump(map_sid_date_r, f)
        
    return map_sid_date_r
    
#loc_file_mkt = "hk_price.txt"
def process_mkt(loc_file_mkt, flag_load_from_tmp_file=False):
    loc_tmp_file = os.path.join(g_loc_tmp, "mkt.tmp.pkl")
    if flag_load_from_tmp_file and os.path.isfile(loc_tmp_file):
        print "load mkt map from", loc_tmp_file
        f = open(loc_tmp_file)
        map_sid_date_p = pickle.load(f)
        return map_sid_date_p
    
    print "process mkd data from", loc_file_mkt
    import datetime
    
    idx_sid = 0
    idx_date = 1
    idx_open = 2
    idx_high = 3
    idx_low = 4
    idx_close = 5
    
    map_sid_date_p = {}
    
    f = open(loc_file_mkt)
    f.readline() #skip the title line
    for line in f:
        items = line.strip().split("\t")
        
        sid = items[idx_sid]
        date = datetime.datetime.strptime(items[idx_date], "%Y-%m-%d")
        p_open = items[idx_open]
        p_high = items[idx_high]
        p_low = items[idx_low]
        p_close = items[idx_close]
        p = (p_open, p_high, p_low, p_close)
        
        if sid in map_sid_date_p:
            map_sid_date_p[sid][date] = p
        else:
            map_sid_date_p[sid] = {date:p}
        
    f.close()
    
    with open(loc_tmp_file, "w") as f:
        pickle.dump(map_sid_date_p, f)
        
    return map_sid_date_p
    
def tokenize(text):
    lst = []
    for i in text:
        lst.append(i)
    return lst
    
#loc_file_doc = "hk_news.txt"
def process_doc(loc_file_doc, flag_load_from_tmp_file=False):
    loc_tmp_file = os.path.join(g_loc_tmp, "doc.tmp.pkl")
    if flag_load_from_tmp_file and os.path.isfile(loc_tmp_file):
        print "load doc map from", loc_tmp_file
        f = open(loc_tmp_file)
        lst_result = pickle.load(f)
        return lst_result
        
    print "process doc from", loc_file_doc
    import datetime
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    idx_date = 0
    idx_sec = 2
    idx_title = 3
    idx_summ = 4
    
    lst_date = []
    lst_doc = []
    lst_lst_sec = []
    f = open(loc_file_doc)
    f.readline()
    for line in f:
        try:
            items = line.strip().split("\t")
            
            date = datetime.datetime.strptime(items[idx_date], "%Y-%m-%d %H:%M:%S")
            lst_date.append(date)
            lst_sec = []
            for s in items[idx_sec].split(","):
                lst_sec.append(s.split("_")[0])
            lst_lst_sec.append(lst_sec)
            doc = items[idx_title].replace(" ", "").strip()
            lst_doc.append(doc)
        except BaseException as e:
            print e
            continue
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfs = tfidf.fit_transform(lst_doc)
    
    lst_result = []
    for idx in xrange(len(lst_date)):
        lst_result.append((lst_date[idx], lst_lst_sec[idx], tfs[idx]))
        
    with open(loc_tmp_file, "w") as f:
        pickle.dump(lst_result, f)
    
    return lst_result
    
def check_label(map_sid_date_p, sid, date, gap):
    if sid not in map_sid_date_p:
        return None
        
    idx_p = 3 #0-open, 1-high, 2-low, 3-close
    
    map_date_p = map_sid_date_p[sid]
    lst_date = sorted(map_date_p.keys()) #ascending
    label = 0
    idx = -1
    for tmp_date in lst_date:
        if tmp_date < date:
            idx +=1
        else:
            break
    
    if idx < len(lst_date)-1-gap and idx>0:
        target_date = lst_date[idx+gap]
        p0 = float(map_date_p[lst_date[idx]][idx_p])
        p1 = float(map_date_p[lst_date[idx+gap]][idx_p])
        if p0*1.01<p1:
            label = 1
        elif p0>p1*1.01:
            label = -1
    else:
        return None
    return label
    
def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    if a is None:
        import copy
        return copy.deepcopy(b)

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a
    
def make_data_slow(loc_data, lst_doc, map_sid_date_p, gap = 2):
    d_data = None
    lst_label = []
    for doc in lst_doc:
        date = doc[0]
        lst_sec = doc[1]
        vec_doc = doc[2]
        for sec in lst_sec:
            label = check_label(map_sid_date_p, sec, date, gap)
            if label is not None:
                d_data = csr_vappend(d_data, vec_doc)
                lst_label.append(label)
    
    print d_data.shape, len(lst_label)
    with open(loc_data, "w") as f_data:
        pickle.dump((d_data, lst_label), f_data)
    
def make_data(loc_data, lst_doc, map_sid_date_p, gap = 2):
    import numpy
    from scipy.sparse import csr_matrix
    
    d_indices = []
    d_indptr = [0]
    d_val = []
    shape_col = 0
    shape_row = -1
    
    lst_label = []
    for doc in lst_doc:
        date = doc[0]
        lst_sec = doc[1]
        vec_doc = doc[2]
        if shape_row>0:
            assert(shape_row==vec_doc.shape[1])
        else:
            shape_row = vec_doc.shape[1]
            
        for sec in lst_sec:
            label = check_label(map_sid_date_p, sec, date, gap)
            if label is not None:
                d_val = numpy.concatenate( [d_val, vec_doc.data] )
                d_indices = numpy.concatenate( [d_indices, vec_doc.indices] )
                d_indptr.append(d_indptr[len(d_indptr)-1] + len(vec_doc.indices))
                lst_label.append(label)
    
                shape_col += 1
                if shape_col%10000==2:
                    print shape_col
                
    print len(lst_label)
    
    d_data = csr_matrix((d_val, d_indices, d_indptr), shape=(shape_col, shape_row))
    
    print d_data.shape
    
    with open(loc_data, "w") as f_data:
        pickle.dump((d_data, lst_label), f_data)

def make_data_paral(idx_process, num_process, lst_doc, map_sid_date_p, reg, gap = 2):
    import numpy
    
    d_indices = []
    d_indptr = [0]
    d_val = []
    shape_col = 0
    shape_row = -1
    
    cnt_idx = 0
    
    lst_label = []
    for doc in lst_doc:
        date = doc[0]
        lst_sec = doc[1]
        vec_doc = doc[2]
        if shape_row>0:
            assert(shape_row==vec_doc.shape[1])
        else:
            shape_row = vec_doc.shape[1]
            
        for sec in lst_sec:
            cnt_idx += 1
            if not cnt_idx%num_process == idx_process:
                continue
        
            label = check_label(map_sid_date_p, sec, date, gap)
            if label is not None:
                d_val = numpy.concatenate( [d_val, vec_doc.data] )
                d_indices = numpy.concatenate( [d_indices, vec_doc.indices] )
                d_indptr.append(d_indptr[len(d_indptr)-1] + len(vec_doc.indices))
                lst_label.append(label)
    
                shape_col += 1
                
    print "PID:", idx_process, len(lst_label)
    
    loc_tmp_file = os.path.join(g_loc_tmp, str(idx_process)+reg)
    import pickle
    with open(loc_tmp_file, "w") as f:
        pickle.dump((d_val, d_indices, d_indptr, shape_col, shape_row, lst_label), f)
    
def make_data_sum(loc_data, reg):
    print "make_data_sum", reg
    
    from scipy.sparse import csr_matrix
    import pickle
    import glob
    import os
    import numpy
    
    d_indices = []
    d_indptr = [0]
    d_val = []
    shape_col = 0
    shape_row = -1
    lst_label = []
    
    lst_file_makedata = glob.glob(os.path.join(g_loc_tmp, reg))
    for loc_f in lst_file_makedata:
        with open(loc_f) as f:
            (tmp_d_val, tmp_d_indices, tmp_d_indptr, tmp_shape_col, tmp_shape_row, tmp_lst_label) = pickle.load(f)
            
            d_val = numpy.concatenate([d_val, tmp_d_val])
            len_d_indices_original = len(d_indices)
            aug_d_indptr = numpy.array(tmp_d_indptr)[1:] + len_d_indices_original
            d_indptr = numpy.concatenate([d_indptr, aug_d_indptr])
            d_indices = numpy.concatenate([d_indices, tmp_d_indices])
            shape_col += tmp_shape_col
            shape_row = max(shape_row, tmp_shape_row)
            assert(shape_row == tmp_shape_row)
            lst_label += tmp_lst_label
            d_data = csr_matrix((d_val, d_indices, d_indptr), shape=(shape_col, shape_row))
            
    print (shape_col, shape_row)
    d_data = csr_matrix((d_val, d_indices, d_indptr), shape=(shape_col, shape_row))
    print d_data.shape
    print len(lst_label)
    
    with open(loc_data, "w") as f_data:
        pickle.dump((d_data, lst_label), f_data)
    
def process_make_data_paral(num_process, lst_doc, map_sid_date_p, gap, loc_data):
    from multiprocessing import Process
    import time
    lst_process = []
    reg = ".makedata.tmp.pkl"
    for idx_process in xrange(num_process):
        p = Process(target=make_data_paral, args=[idx_process, num_process, lst_doc, map_sid_date_p, reg, gap])
        p.start()
        start_time = time.time()
        lst_process.append((p, start_time))
        
    for pp in lst_process:
        pp[0].join()
        print pp[0], str(time.time() - pp[1])+" seconds"
    
    print "reduce"
    make_data_sum(loc_data, '*'+reg)
    
def split_train_test_slow(loc_data, loc_train, loc_test):
    with open(loc_data) as f:
        (d_data, lst_label) = pickle.load(f)
        
    d_train = None
    l_train = []
    d_test = None
    l_test = []
    from random import shuffle
    cho = [i for i in xrange(d_data.shape[0])]
    shuffle(cho)
    for i in xrange(d_data.shape[0]):
        tmp = d_data.getrow[i]
        label = lst_label[i]
        if cho[i]>len(cho)*0.2: #train
            d_train = csr_vappend(d_train, tmp)
            l_train.append(label)
        else:#test
            d_test = csr_vappend(d_test, tmp)
            l_test.append(label)
        
    with open(loc_train, "w") as f:
        pickle.dump((d_train, l_train), f)

    with open(loc_test, "w") as f:
        pickle.dump((d_test, l_test), f)
        
def split_train_test(loc_data, loc_train, loc_test):
    print "split"
    with open(loc_data) as f:
        (d_data, lst_label) = pickle.load(f)

    d_val = d_data.data
    d_indices = d_data.indices
    d_indptr = d_data.indptr

    train_d_indices = []
    train_d_indptr = [0]
    train_d_val = []
    train_shape_col = 0
    train_shape_row = d_data.shape[1]

    test_d_indices = []
    test_d_indptr = [0]
    test_d_val = []
    test_shape_col = 0
    test_shape_row = d_data.shape[1]



    l_train = []
    l_test = []

    import numpy
    from scipy.sparse import csr_matrix
    from random import shuffle
    cho = [i for i in xrange(d_data.shape[0])]
    shuffle(cho)

    for i in xrange(1, len(d_indptr)-1):
        if i%10000==1:
            print i

        tmp_d_indices = d_indices[d_indptr[i-1]:d_indptr[i]]
        tmp_d_val = d_val[d_indptr[i-1]:d_indptr[i]]

        label = lst_label[i]
        if cho[i]>len(cho)*0.2: #train
            train_shape_col += 1
            train_d_val = numpy.concatenate( [train_d_val, tmp_d_val] )
            train_d_indices = numpy.concatenate( [train_d_indices, tmp_d_indices] )
            train_d_indptr.append(train_d_indptr[len(train_d_indptr)-1] + len(tmp_d_indices))
            l_train.append(label)
        else:#test
            test_shape_col += 1
            test_d_val = numpy.concatenate( [test_d_val, tmp_d_val] )
            test_d_indices = numpy.concatenate( [test_d_indices, tmp_d_indices] )
            test_d_indptr.append(test_d_indptr[len(test_d_indptr)-1] + len(tmp_d_indices))
            l_test.append(label)


    d_train = csr_matrix((train_d_val, train_d_indices, train_d_indptr), shape=(train_shape_col, train_shape_row))
    d_test = csr_matrix((test_d_val, test_d_indices, test_d_indptr), shape=(test_shape_col, test_shape_row))

    with open(loc_train, "w") as f:
        pickle.dump((d_train, l_train), f)

    with open(loc_test, "w") as f:
        pickle.dump((d_test, l_test), f)

        
def split_train_test_paral(idx_process, num_process, cho, d_data, lst_label, reg_train, reg_test):
    import numpy
    from scipy.sparse import csr_matrix
    
    d_val = d_data.data
    d_indices = d_data.indices
    d_indptr = d_data.indptr
    
    
    train_d_indices = []
    train_d_indptr = [0]
    train_d_val = []
    train_shape_col = 0
    train_shape_row = d_data.shape[1]

    test_d_indices = []
    test_d_indptr = [0]
    test_d_val = []
    test_shape_col = 0
    test_shape_row = d_data.shape[1]

    l_train = []
    l_test = []
    
    for i in xrange(1, len(d_indptr)-1):
        if not i%num_process == idx_process:
            continue
            
        tmp_d_indices = d_indices[d_indptr[i-1]:d_indptr[i]]
        tmp_d_val = d_val[d_indptr[i-1]:d_indptr[i]]
        
        label = lst_label[i]
        if cho[i]>len(cho)*0.2: #train
            train_shape_col += 1
            train_d_val = numpy.concatenate( [train_d_val, tmp_d_val] )
            train_d_indices = numpy.concatenate( [train_d_indices, tmp_d_indices] )
            train_d_indptr.append(train_d_indptr[len(train_d_indptr)-1] + len(tmp_d_indices))
            l_train.append(label)
        else:#test
            test_shape_col += 1
            test_d_val = numpy.concatenate( [test_d_val, tmp_d_val] )
            test_d_indices = numpy.concatenate( [test_d_indices, tmp_d_indices] )
            test_d_indptr.append(test_d_indptr[len(test_d_indptr)-1] + len(tmp_d_indices))
            l_test.append(label)
            
    loc_tmp_file = os.path.join(g_loc_tmp, str(idx_process)+reg_train)
    import pickle
    with open(loc_tmp_file, "w") as f:
        pickle.dump((train_d_val, train_d_indices, train_d_indptr, train_shape_col, train_shape_row, l_train), f)

    loc_tmp_file = os.path.join(g_loc_tmp, str(idx_process)+reg_test)
    import pickle
    with open(loc_tmp_file, "w") as f:
        pickle.dump((test_d_val, test_d_indices, test_d_indptr, test_shape_col, test_shape_row, l_test), f)
        
def process_split_train_test_paral(num_process, loc_data, loc_train, loc_test):
    print "split"
    with open(loc_data) as f:
        (d_data, lst_label) = pickle.load(f)
        
    
    from random import shuffle
    cho = [i for i in xrange(d_data.shape[0])]
    shuffle(cho)
    
    from multiprocessing import Process
    import time
    lst_process = []
    reg_train = ".train.split.pkl"
    reg_test = ".test.split.pkl"
    
    for idx_process in xrange(num_process):
        p = Process(target=split_train_test_paral, args=[idx_process, num_process, cho, d_data, lst_label, reg_train, reg_test])
        p.start()
        start_time = time.time()
        lst_process.append((p, start_time))
        
    for pp in lst_process:
        pp[0].join()
        print pp[0], str(time.time() - pp[1])+" seconds"
    
    print "reduce"
    make_data_sum(loc_train, '*'+reg_train)
    make_data_sum(loc_test, '*'+reg_test)
    
        
def adhoc_eval_research(map_sid_date_r, map_sid_date_p):
    cnt_all = 0
    cnt_true2 = 0
    cnt_true5 = 0
    cnt_true10 = 0
    
    cnt_pos2 = [0,0]
    cnt_neg2 = [0,0]
    cnt_neu2 = [0,0]
    cnt_pos5 = [0,0]
    cnt_neg5 = [0,0]
    cnt_neu5 = [0,0]
    cnt_pos10 = [0,0]
    cnt_neg10 = [0,0]
    cnt_neu10 = [0,0]
    
    import datetime
    date_min = datetime.datetime.strptime("2016-12-12 12:12:12", "%Y-%m-%d %H:%M:%S")
    date_max = datetime.datetime.strptime("1880-12-12 12:12:12", "%Y-%m-%d %H:%M:%S")
    for sid in map_sid_date_r:
        lst_research = map_sid_date_r[sid]
        for (date, label, agent) in lst_research:
            if date<date_min:
                date_min = date
            if date>date_max:
                date_max = date
        
            cnt_all += 1
            
            # l2 = check_label(map_sid_date_p, sid, date, gap=20)
            # l5 = check_label(map_sid_date_p, sid, date, gap=60)
            # l10 = check_label(map_sid_date_p, sid, date, gap=120)
            
            # if l2 is not None and l2==label:
                # cnt_true2 += 1
                
            # if l5 is not None and l5==label:
                # cnt_true5 += 1
                
            # if l10 is not None and l10==label:
                # cnt_true10 += 1
                
            
            # if l2>0:
                # cnt_pos2[0] += 1
            # elif l2<0:
                # cnt_neg2[0] += 1
            # else:
                # cnt_neu2[0] += 1
                
            # if l5>0:
                # cnt_pos5[0] += 1
            # elif l5<0:
                # cnt_neg5[0] += 1
            # else:
                # cnt_neu5[0] += 1
                
            # if l10>0:
                # cnt_pos10[0] += 1
            # elif l10<0:
                # cnt_neg10[0] += 1
            # else:
                # cnt_neu10[0] += 1
                
            
            # if label>0:
                # cnt_pos2[1] += 1
                # cnt_pos5[1] += 1
                # cnt_pos10[1] += 1
            # elif label<0:
                # cnt_neg2[1] += 1
                # cnt_neg5[1] += 1
                # cnt_neg10[1] += 1
            # else:
                # cnt_neu2[1] += 1
                # cnt_neu5[1] += 1
                # cnt_neu10[1] += 1
                
            # if cnt_all % 5000==2:
                # print cnt_all
                
                
    # print "2", cnt_true2 * 1.0 / cnt_all, cnt_pos2, cnt_neg2, cnt_neu2
    # print "5", cnt_true5 * 1.0 / cnt_all, cnt_pos5, cnt_neg5, cnt_neu5
    # print "10", cnt_true10 * 1.0 / cnt_all, cnt_pos10, cnt_neg10, cnt_neu10
    
    print "all", cnt_all
    print date_min
    print date_max

if __name__ == '__main__':
    loc_file_mkt = os.path.join(g_loc_root, "hk_price.txt")
    map_sid_date_p = process_mkt(loc_file_mkt, flag_load_from_tmp_file=True)
    
    loc_file_doc = os.path.join(g_loc_root, "hk_news.txt")
    lst_doc = process_doc(loc_file_doc, flag_load_from_tmp_file=True)
    
    loc_data  = os.path.join(g_loc_tmp, "data.pkl")
    ################
    process_make_data_paral(30, lst_doc, map_sid_date_p, gap=2, loc_data)
    ################
    # make_data(loc_data, lst_doc, map_sid_date_p, gap = 2)
    ################
    
    process_split_train_test_paral(30, loc_data, g_loc_train, g_loc_test)
    
    # map_sid_date_p = None
    # loc_file_research = os.path.join(g_loc_root, "hk_research.txt")
    # map_sid_date_r = process_research(loc_file_research, flag_load_from_tmp_file=False)
    # adhoc_eval_research(map_sid_date_r, map_sid_date_p)
    
    # loc_file_doc = os.path.join(g_loc_root, "hk_news.txt")
    # ad_hoc_process_doc(loc_file_doc)