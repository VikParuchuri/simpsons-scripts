setwd("~/vikparuchuri/simpsons-scripts")

is_installed <- function(mypkg) is.element(mypkg, installed.packages()[,1])

load_or_install<-function(package_names)
{
  for(package_name in package_names)
  {
    if(!is_installed(package_name))
    {
      install.packages(package_name,repos="http://lib.stat.cmu.edu/R/CRAN")
    }
    options(java.parameters = "-Xmx8g")
    library(package_name,character.only=TRUE,quietly=TRUE,verbose=FALSE)
  }
}

load_or_install(c("RJSONIO","ggplot2","stringr","foreach","wordcloud","lsa","MASS","openNLP","tm","fastmatch","reshape","openNLPmodels.en",'e1071','gridExtra'))


ad = read.csv('full_results.csv',row.names=1,stringsAsFactors=FALSE)

characters = unique(ad$label)

characters = tapply(ad$result_label,ad$result_label,length)
characters = names(characters)[characters>20]
characters = characters[!characters %in% c("", "Tertiary","Patty","Troy","Dr.Nick")]

afinn_list<-read.delim(file="AFINN-111.txt",header=FALSE,stringsAsFactors=FALSE)
names(afinn_list)<-c("word","score")
afinn_list$word<-tolower(afinn_list$word)

full_term_list<-afinn_list$word

all_score_frames<-list()
ri_cols<-1000

ad$line = iconv(ad$line,"UTF-8", "ASCII")

neg_vec = colSums(do.call(rbind,lapply(which(afinn_list$score< -2),function(x){
  set.seed(x)
  sample_vec<-rep(0,ri_cols)
  s_inds<-sample(1:length(sample_vec),5)
  sample_vec[s_inds]<-1
  sample_vec
})))

pos_vec = colSums(do.call(rbind,lapply(which(afinn_list$score > 2),function(x){
  set.seed(x)
  sample_vec<-rep(0,ri_cols)
  s_inds<-sample(1:length(sample_vec),5)
  sample_vec[s_inds]<-1
  sample_vec
})))

for(z in 1:length(characters))
{
  ppatterns<-c("\\n","\\r")
  sel_data = lapply(2:(nrow(ad)-1),function(x){
    ret<-NA
    if(ad$result_label[x]==characters[z]){
      start = x-1
      iscs = characters[z]
      while(start>2){
        iscs = ad$result_label[start]
        if(iscs!=characters[z]){
          break
        }
        start = start-1
      }
      end= x+1
      isce = characters[z]
      while(end<(nrow(ad)-1)){
        isce = ad$result_label[end]
        if(isce!=characters[z]){
          break
        }
        end = end+1
      }
      isc = unique(c(iscs,isce))
      isc = isc[isc %in% characters]
      isc = isc[isc!=characters[z]]
      if(length(isc)>0){
        ret = data.frame(isc,char=rep(characters[z],length(isc)),line=rep(ad$line[x],length(isc)),stringsAsFactors=FALSE)
      }
    }
    ret
  })
  sel_data = sel_data[!is.na(sel_data)]
  sel_frame = do.call(rbind,sel_data)
  combined<-tolower(gsub(paste("(",paste(ppatterns,collapse="|"),")",sep=""),"",sel_frame$line))
  tokenized_combined<-lapply(combined,scan_tokenizer)
  ri_mat<-matrix(0,nrow=length(characters),ncol=ri_cols)
  rownames(ri_mat)<-characters
  
  if(length(combined)>1){
    for(i in 1:length(combined))
    {
      if(i%%10000==0)
        print(i)
      tokens<-tokenized_combined[[i]]
      tokens<-tokens[tokens %in% full_term_list]
      if(length(tokens)>0){
        for(tok in tokens){
          set.seed(which(full_term_list==tok)[1])
          sample_vec<-rep(0,ri_cols)
          s_inds<-sample(1:length(sample_vec),5)
          sample_vec[s_inds]<-1
          ri_mat[sel_frame[i,'isc'],]<-ri_mat[sel_frame[i,'isc'],]+sample_vec
        }
      }
    }
    
    set.seed(1)
    neg_scores = as.numeric(apply(ri_mat,1,function(x) {
      cosine(x,neg_vec)
    }))
    pos_scores = as.numeric(apply(ri_mat,1,function(x) {
      cosine(x,pos_vec)
    }))
    score_frame<-data.frame(character=rownames(ri_mat),pos_scores,neg_scores,score=pos_scores-neg_scores,stringsAsFactors=FALSE)
    sorted_score_frame<-score_frame[order(score_frame$score),]
    all_score_frames[[z]]<-sorted_score_frame
  }
}

for(z in 1:length(characters)){
  char = characters[z]
  if(!is.null(all_score_frames[[z]])){
    ssf = all_score_frames[[z]][!is.na(all_score_frames[[z]]$score),]
    c <- ggplot(ssf, aes(x=character,y=score))
    c = c + geom_bar() + labs(title=paste("How",char,"feels about the others",sep=" "),x="Character",y="Sentiment towards")
    c = c + theme(legend.position="none",
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank()
    )
    print(c)
  }
}

total_frame = data.frame(matrix(0,nrow=length(characters),ncol=length(characters)))
colnames(total_frame) = characters
rownames(total_frame) = characters

for(i in 1:nrow(total_frame)){
  char = characters[i]
  if(!is.null(all_score_frames[[i]])){
    ssf = all_score_frames[[i]][!is.na(all_score_frames[[i]]$score),]
    total_frame[char,ssf$character] = ssf$score
  }
}

col = brewer.pal(ncol(total_frame),"RdYlGn")

heatmap(as.matrix(total_frame),Rowv=NA, Colv=NA, col = col, scale="column", margins=c(5,10),ColSideColors=col,main="Simpsons Character Sentiments")

term<-c("merkel","blair","jintao","ahmadinejad","chavez")
term_vec<-foreach(i=1:length(all_score_frames),.combine=rbind) %do%
{
  score_row<-rep(0,length(term))
  for(z in 1:length(score_row))
  {
    sel_score<-all_score_frames[[i]][all_score_frames[[i]]$term==term[z],"score"]
    sel_score[is.na(sel_score)]<-0
    if(length(sel_score)==0)
      sel_score<-0
    score_row[z]<-round(sel_score,5)
  }
  as.numeric(c(date_max_list[i],score_row))
}
term_vec<-as.data.frame(term_vec)
names(term_vec)<-c("year",term)

term_df <- melt(term_vec, id.vars="year")
term_means<-sapply(all_score_frames,function(x) mean(x$score))

text_size<-40
ggplot(data=term_df,aes(x=year, y=value, colour=variable))+geom_line(size=1) + geom_line(aes(x = as.numeric(date_max_list), y = term_means), colour = "black",size=1.5) + ylab("sentiment") + opts(title = expression("US Sentiment (+/-) Over Time"),legend.text=theme_text(size=text_size),legend.title=theme_text(size=0),plot.title=theme_text(size=text_size),axis.text.y=theme_text(size=text_size),axis.text.x=theme_text(size=text_size),axis.title.y=theme_text(size=text_size,angle=90),axis.title.x=theme_text(size=text_size),legend.key.size=unit(2,"cm")) 


qplot(x=term_vec$year,term_vec$score,geom="line")

qplot(x=as.numeric(date_max_list),y=term_means,geom="line")

corpus<-Corpus(VectorSource(combined))
corpus<-tm_map(corpus,stripWhitespace)
overall_matrix<-as.matrix(TermDocumentMatrix(corpus,control=list(weighting=weightBin,removePunctuation=TRUE,removeNumbers=TRUE,stopwords=TRUE,wordLengths=c(4,20),bounds=list(local = c(2,Inf)))))
overall_matrix<-overall_matrix[rownames(overall_matrix) %in% c(jrc_names,afinn_list$word),]

space<-lsa(t(overall_matrix))
space_sk_inv<-ginv(diag(space$sk))
transform_space_tk<-t(space$tk)
space_sk_diag<-diag(space$sk)
lsa_doc_vecs<-apply(space$dk,1,function(x) x %*% space_sk_diag)

afinn_cols<-which(rownames(overall_matrix) %in% afinn_list$word)

