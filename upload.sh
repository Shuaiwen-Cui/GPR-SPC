# try "sh upload.sh" or "bash upload.sh" or ".\upload.sh" or "./upload.sh" to run this script

echo '--------upload files start--------'
# enter the target folder
# cd ./

# git init
git add .
git status
# git commit -m "auto commit by upload.sh"
git commit -m 'auto commit by upload.sh'
echo '--------commit successfully--------'

# git push -f https://github.com/Shuaiwen-Cui/GPR-SPC.git main
git push -u https://github.com/Shuaiwen-Cui/GPR-SPC.git main
# git remote add origin https://github.com/Shuaiwen-Cui/GPR-SPC.git
# git push -u origin main
echo '--------push to GitHub successfully--------'

# git push -f <url> master
# git push -u <url> master
# git remote add origin <url>
# git push -u origin master
# echo '--------push to Gitee successfully--------'

# if to deploy to https://<USERNAME>.github.io/<REPO>
# git push -f git@github.com:<USERNAME>/<REPO>.git master:gh-pages
# done

# if authentication required, username is your GitHub username and password is your GitHub password (deprecated) or personal access token (recommended).

