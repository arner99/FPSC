input="databases_cog_clone.txt"
while IFS= read -r line
do
  git clone https://github.com/lexibank/"$line"
  echo "database $line cloned successfully."
done < "$input"