while read line
do
  line=${line%?}
  curl --location --request GET "cs7ns1.scss.tcd.ie?shortname=dengji&myfilename=${line}" > "${line}"
done < files.txt