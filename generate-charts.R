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

transcripts = fromJSON("data/transcripts.json")

trans = lapply(transcripts, function(x) {
  s = x[['script']]
  s = iconv(s,"UTF-8", "ASCII")
  s = gsub(" +"," ",s)
  s = gsub("[\r\t]","",s)
  tolower(s)
}
)

all_text = paste(trans,collapse="\n")
line_count = str_count(all_text,"\n")

word_count = str_count(all_text," ")

initial_data = fromJSON("data/final_voice.json")
#speakers = unlist(lapply(initial_data,function(x) lapply(x,function(y) y[['speaker']])))
#lines = unlist(lapply(initial_data,function(x) lapply(x,function(y) y[['line']])))
speakers = unlist(lapply(initial_data,function(x)  x[['speaker']]))
lines = unlist(lapply(initial_data,function(x) x[['line']]))


voice_data = data.frame(speaker = speakers, line = lines)
voice_data = voice_data[voice_data$speaker!="",]

tab = table(voice_data$speaker)

m = qplot(names(tab), as.numeric(tab),geom="histogram")
m = m + ylab("Number of Lines") + xlab("Character Name") + ggtitle("Lines per Character in Scripts")
m

unique_speakers = unique(voice_data$speaker)
all_score_frames<-foreach(z=1:length(unique_speakers)) %do%
{
  combined<-sentDetect(tolower(voice_data$line[voice_data$speaker==unique_speakers[z]]))
  comb_words<-lapply(combined,function(x){ ret<-gsub("[^A-Za-z]","",scan_tokenizer(x))
                                           ret<-ret[nchar(ret)>3 & nchar(ret)<20]
                                           ret
  })
  comb_words
}

thresh = 4
sel_df<-foreach(i=1:length(unique_speakers)) %do%
{
  thresh<-0
  all_sc<-unlist(all_score_frames[[i]])
  all_sc<-all_sc[!all_sc %in% c(stopwords("en"))]
  speaker_tab = table(all_sc)
  all_sc<-all_sc[all_sc %in% names(speaker_tab)[speaker_tab>thresh]]
  all_sc<-data.frame(word=names(table(all_sc)),freq=as.numeric(table(all_sc)),stringsAsFactors=FALSE)
  all_sc<-data.frame(all_sc[order(all_sc$freq,decreasing=TRUE),],pos=1:nrow(all_sc),speaker=unique_speakers[i])
  if(nrow(all_sc)>7) {
    all_sc<-all_sc[1:7,]
  }
  all_sc
}

comb_df = do.call(rbind,sel_df)
comb_df = comb_df[order(comb_df$speaker, increasing=TRUE)]

p <- ggplot(comb_df, aes(x=speaker, y=pos, label=word,color=freq,size=freq)) 
p<-p + geom_text() + scale_size(range = c(7, 12), name="Word Frequency",guide="none")+scale_colour_gradient(low="brown", high="darkblue",guide="none") +xlab("")+ylab("")+theme_bw()
p<-p+theme(panel.grid.major=theme_blank(),panel.grid.minor=element_blank(), plot.title=theme_text(size=30), axis.text.x  = theme_text(size=20),axis.ticks = element_blank(), axis.text.y = element_blank())
p<-p+scale_x_discrete(expand=c(.2,.1)) + opts(axis.line = element_line())
p

s = 800
a =s:(s+5)
paste(voice_data[a,'speaker'],": ",voice_data[a,'line'])

ad = read.csv('full_results.csv',row.names=1,stringsAsFactors=FALSE)

feature_names = names(ad)[7:(length(names(ad))-3)]
labelled_data = ad[ad[,'label']!='',]

tf = ad
scaled_data = scale(tf[,feature_names])
scaled_data = apply(scaled_data,2,function(x) {
  x[is.na(x)] = -1
  x
})
svd_train<-svd(scaled_data,2)$u

newtrain<-data.frame(x=svd_train[,1],y=svd_train[,2],score=as.factor(tf$result_code))

#model = svm(score ~ x + y, data = newtrain)
#plot(model,newtrain)

collapse_frame = do.call(rbind,by(tf[,feature_names],tf$result_label,function(x) apply(x,2,mean)))
line_count = tapply(tf$result_label,tf$result_label,length)
scaled_data = scale(collapse_frame)
scaled_data = apply(scaled_data,2,function(x) {
  x[is.na(x)] = -1
  x
})
svd_train<-data.frame(svd(scaled_data,2)$u,line_count=line_count,label=rownames(line_count))
svd_train <- svd_train[svd_train$X1<mean(svd_train$X1)+1.4*sd(svd_train$X1) & svd_train$X1>mean(svd_train$X1)-1.4*sd(svd_train$X1),]
svd_train <- svd_train[svd_train$X2<mean(svd_train$X2)+1.4*sd(svd_train$X2) & svd_train$X2>mean(svd_train$X2)-1.4*sd(svd_train$X2),]
p <- ggplot(svd_train, aes(X1, X2))
p = p + geom_point(aes(colour = svd_train$label,size = log(svd_train$line_count))) + scale_size_area(max_size=20) + geom_text(data = svd_train[svd_train$line_count>100,], aes(X1,X2, label = label), hjust = 2)
p = p +   theme(axis.line = element_blank(),
               panel.grid.major = element_blank(),
               panel.grid.minor = element_blank(),
               panel.border = element_blank(),
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                axis.ticks=element_blank(),
                axis.text.x = element_blank(),
                axis.text.y = element_blank()) 
p = p +labs(colour="Character",size="Number of Lines")
p

sel = ad[,c('line','label','result_label','season','episode')]

labelled_data = ad[ad[,'label']!='' & ad[,'result_code']!=-1,]
correct_label = as.numeric(lapply(1:nrow(labelled_data),function(x){
    all_labels = labelled_data$label[(x-1):(x+1)]
    ret = FALSE
    if(labelled_data$result_label[x] %in% all_labels){
      ret = TRUE
    }
    ret
}))

exact_correct = labelled_data$label==labelled_data$result_label
sum(correct_label)/nrow(labelled_data)
sum(exact_correct)/nrow(labelled_data)

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
    score_frame<-data.frame(character=rownames(ri_mat),pos_scores,neg_scores,score=pos_scores-neg_scores)
    sorted_score_frame<-score_frame[order(score_frame$score),]
    all_score_frames[[z]]<-sorted_score_frame
  }
}

for(z in 1:length(characters)){
  char = characters[z]
  if(!is.null(all_score_frames[[z]])){
    ssf = all_score_frames[[z]][!is.na(all_score_frames[[z]]$score),]
    c <- ggplot(ssf, aes(x=character,y=score),fill=(character))
    c = c + geom_bar() + labs(title=paste("How",char,"feels about the rest",sep=" "))
    print(c)
  }
}

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


