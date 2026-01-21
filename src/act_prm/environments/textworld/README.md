# TextWorld Environment

## Setup

Following [BALROG](https://github.com/balrog-ai/BALROG/tree/main/balrog/environments/textworld) and [Intelligent Go-Explore](https://github.com/conglu1997/intelligent-go-explore/tree/main), we use pre-generated games for the `coin_collector`, `the_cooking_game`, and `treasure_hunter` tasks.

First create a directory for the game files (which should be the same as `dataset_config.cache_dir` in the `configs/environments/textworld/*.yaml` environment config). 

Then, download the game files from [https://github.com/conglu1997/intelligent-go-explore/tree/main/textworld/tw_games](https://github.com/conglu1997/intelligent-go-explore/tree/main/textworld/tw_games). BALROG provides a convenient Google Drive link, so we can set things up in a single command:

```bash
mkdir /scr/mzhang/data/textworld && cd /scr/mzhang/data/textworld  # change to your own directory
curl -L -o tw-games.zip 'https://drive.google.com/uc?export=download&id=1aeT-45-OBxiHzD9Xn99E5OvC86XmqhzA'
unzip tw-games.zip
```
