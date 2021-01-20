# DocBot-Server


DocBot is a crawler which scans legal agreement for a specific set of keywords to match with ToS;DR Cases.



## Installation

1. `npm install`
2. Create an `.env` file from `.env.example`
3. Change ports if necessary, otherwise set only an API Key

## Usage

1. Turn on the server with `npm run start`. This will boot up both gateway and http server on the defined ports
2. Go to `https://your_installation:6004`
3. Set the IP and API Key of your instance, enter a service id from edit.tosdr.org and hit "Crawl"
4. Now every match should be listed
