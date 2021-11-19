# DocBot-Server

[![Build Status](https://ci.git.tosdr.org/api/badges/tosdr/DocBot-Server/status.svg)](https://ci.git.tosdr.org/tosdr/DocBot-Server)

DocBot is a crawler which scans legal agreement for a specific set of keywords to match with ToS;DR Cases.



## Installation

1. `npm install`
2. Create an `.env` file from `.env.example`
3. Change ports if necessary, otherwise set only an API Key

## Usage

1. Turn on the server with `npm run start`. This will boot up both gateway and http server on the defined ports
2. Go to `https://your_installation:6004`
3. Set the gateway server (for example `ws://0.0.0.0:6005`) and the API Key used to connect to Crisp. Hit "Refresh Cases" which gets a list of regex from the database.
4. Enter a service id from edit.tosdr.org and hit "Crawl". Now every match should be listed

API calls are served from [Crisp](https://github.com/tosdr/CrispCMS). You can call a local instance by modifying `.env`.
