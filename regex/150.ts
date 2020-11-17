import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*spidering)|(?=.*spider)|(?=.*crawler)|(?=.*crawling))((?=.*not)|(?=.*no))((?=.*automatic)|(?=.*automation))"),
	caseID: 150
} as Regex;