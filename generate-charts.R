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

load_or_install(c("RJSONIO","ggplot2","stringr"))

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