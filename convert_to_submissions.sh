mkdir -p submissions


counter=500

for system in Aya23 CommandR-plus DeepL GoogleTranslate Llama3-70B MicrosoftTranslator YandexTranslate Gemini-1.5-Pro Mistral-Large Claude-3.5 GPT-4 Phi-3-Medium; do 

    systemname=$system
    if [ $system == "MicrosoftTranslator" ]; then
        systemname="ONLINE"
    fi
    if [ $system == "GoogleTranslate" ]; then
        systemname="ONLINE"
    fi
    if [ $system == "YandexTranslate" ]; then
        systemname="ONLINE"
    fi
    if [ $system == "DeepL" ]; then
        systemname="ONLINE"
    fi

    echo '{
     "institution_name": "'$systemname'",
     "name": "'$systemname'",
     "publication_name": "'$systemname'",
     "primary_submissions": [' >> submissions/teams.json

    for file in `ls wmt_translations/$system/ --color=no | grep "xml.full" | grep -v tokens`; do 
        name=`echo $file | sed 's/xml.full.*/xml/'`
        sourcefile="wmt_testset/"$name
        outputfile="submissions/"`echo $name | sed "s/xml/$systemname.$counter.xml/"`
        src=`echo $file | sed 's/.*\.//'`
        trg=`echo $file | sed 's/.xml.*//' | sed 's/.*-//'`
        testsetname=`echo $name | sed 's/.xml//' | sed 's/wmt_translations.//'`

        echo $system $file $trg $sourcefile $outputfile
        
        wmt-wrap -s $sourcefile -t wmt_translations/$system/$file -n $systemname -l $trg > $outputfile

        counter=$((counter+1))
        # if systemname is not Llama3-70B or Aya23
        if [ $systemname != "Llama3-70B" ] && [ $systemname != "Aya23" ]; then
            echo '{
                "competition": "WMT24: General MT Task",
                "file_name": "'$outputfile'",
                "is_constrained": false,
                "is_open_source": false,
                "is_primary": true,
                "is_removed": false,
                "language_pair": "'$src'-'$trg'",
                "score": 0,
                "score_chrf": 0,
                "submission_id": '$counter',
                "test_set": "'$testsetname'"
            },' >> submissions/teams.json
        else
            echo '{
                "competition": "WMT24: General MT Task",
                "file_name": "'$outputfile'",
                "is_constrained": false,
                "is_open_source": true,
                "is_primary": true,
                "is_removed": false,
                "language_pair": "'$src'-'$trg'",
                "score": 0,
                "score_chrf": 0,
                "submission_id": '$counter',
                "test_set": "'$testsetname'"
            },' >> submissions/teams.json
        fi
    done

    echo ']
    },' >> submissions/teams.json
done
