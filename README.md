# cobra
An artificial intelligence project, aimed at using a Keras Neural Network to predict car make by a photo

## Git Flow

The following is an outline of the steps we'll need to take to collaborate seamlessly on the project.

### Contributing
1. We will have 2 main branches: `develop`, and `master`.
2. When you go to add a feature, branch off of the latest `develop`. A good convention to use would be firstnameinit.feature-description. For example, George Boole might create a branch named `georgeb.login-styling-fix`
3. After you have finished that feature, create a pull request. After tagging the PR apropriately, choose one or more reviewers who know that part of the system well. Wait for their approval before merging the PR. If they have comments or merge conflicts, address those first and wait for approval. If you are addressing a specific issue, #reference that issue in the PR description
4. After merging and confirming, close any issues that the branch fixes.

### Issues
1. As we find bugs or identify new features, we'll create issues.
2. When you create an issue, apply the appropriate tags to it.
3. To take responsibility for fixing an issue, assign it to yourself before beginning work on it.

### Releasing
1. As we move on, the software will become more complete. Here, we have a chance to make sure it is ready for prime time.
2. Only once we are certain it is ready, the code is merged to `master`. `master` will always contain only stable code

### Git refresher
These are some of the git commands you'll use a lot. They are listed here for reference. If you'd like to get more comfortable with Git, check out this guide from Atlassian: https://www.atlassian.com/git/tutorials

`git clone <repourl>`  
clone a remote repository (like this one) to your local

`git pull`  
Move to the latest commit of the current branch

`git branch`  
view all the branches on your local

`git branch <branchname>`  
create a new branch. Note that you are NOT automatically switched to that branch

`git checkout <branchname>`  
Checkout an existing branch

`git fetch` `git merge`  
This sequence will update your current branch and pull down any other branches from the remote repository

`git stash`  
Place ALL of the changes since your last commit onto a stack. This is a great way of undoing a bunch of changes

`git stash pop`  
pops changes back off the stack. This is a great way of redoing all those things you just un-did with git stash
