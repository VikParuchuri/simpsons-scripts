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

ad = read.csv('audio_data.csv',row.names=1,stringsAsFactors=FALSE)

feature_names = names(ad)[7:(length(names(ad))-3)]
labelled_data = ad[ad[,'label']!='',]

tf = ad
scaled_data = scale(tf[,feature_names])
scaled_data = apply(scaled_data,2,function(x) {
  x[is.na(x)] = -1
  x
})
svd_train<-svd(scaled_data,2)$u

newtrain<-data.frame(x=svd_train[,1],y=svd_train[,2],score=as.factor(tf$label_code))

model = svm(score ~ x + y, data = newtrain)
plot(model,newtrain)

collapse_frame = do.call(rbind,by(tf[,feature_names],tf$label,function(x) apply(x,2,mean)))
line_count = tapply(tf$label,tf$label,length)
scaled_data = scale(collapse_frame)
scaled_data = apply(scaled_data,2,function(x) {
  x[is.na(x)] = -1
  x
})
svd_train<-data.frame(svd(scaled_data,2)$u,line_count=line_count,label=rownames(line_count))
svd_train <- svd_train[svd_train$X1<mean(svd_train$X1)+1.4*sd(svd_train$X1) & svd_train$X1>mean(svd_train$X1)-1.4*sd(svd_train$X1),]
svd_train <- svd_train[svd_train$X2<mean(svd_train$X2)+1.4*sd(svd_train$X2) & svd_train$X2>mean(svd_train$X2)-1.4*sd(svd_train$X2),]
p <- ggplot(svd_train, aes(X1, X2))
p = p + geom_point(aes(colour = svd_train$label,size = svd_train$line_count)) + scale_size_area(max_size=20) + geom_text(data = svd_train[svd_train$line_count>10,], aes(X1,X2, label = label), hjust = 2)
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
grid.table(sel)

correct_label = as.numeric(lapply(1:nrow(labelled_data),function(x){
    all_labels = labelled_data$label[(x-1):(x+1)]
    ret = FALSE
    if(labelled_data$result_label[x] %in% all_labels){
      ret = TRUE
    }
    ret
}))

exact_correct = labelled_data$label==labelled_data$result_label