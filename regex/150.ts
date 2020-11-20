import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*spidering)|(?=.*spider)|(?=.*crawler)|(?=.*crawling))((?=.*not)|(?=.*no))((?=.*automatic)|(?=.*automation))", "i"),
	caseID: 150
} as Regex;