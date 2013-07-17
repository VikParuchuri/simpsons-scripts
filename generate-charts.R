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

load_or_install(c("RJSONIO","ggplot2","stringr","foreach","wordcloud","lsa","MASS","openNLP","tm","fastmatch","reshape","openNLPmodels.en"))

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

initial_data = fromJSON("data/initial_voice.json")
speakers = unlist(lapply(initial_data,function(x) lapply(x,function(y) y[['speaker']])))
lines = unlist(lapply(initial_data,function(x) lapply(x,function(y) y[['line']])))

voice_data = data.frame(speaker = speakers, line = lines)

tab = table(voice_data$speaker)

m = qplot(names(tab), as.numeric(tab),geom="histogram")
m = m + ylab("Number of Lines") + xlab("Character Name") + ggtitle("Lines per Character in Scripts")
m

unique_speakers = unique(speakers)
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
ggsave(plot=p,filename=paste("new/",country_list[cou],".png",sep=""),width=15,height=15)

