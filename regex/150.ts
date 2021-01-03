import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*spidering)|(?=.*spider)|(?=.*crawler)|(?=.*crawling))((?=.*not)|(?=.*no))|((?=.*not)|(?=.*no))((?=.*automatic)|(?=.*automation)|(?=.*automate)(?=.*access))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 150,
	name: "Spidering or crawling is not allowed"
} as Regex;