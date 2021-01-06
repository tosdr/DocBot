import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*visited )((?=.*prior)|(?=.*before))|(?=.*visited from)(?=.*domain)|((?=.*referrer)|(?=.*referral)|(?=.*referring)|(?=.*referred))((?=.*url)|(?=.*http)|(?=.*website)|(?=.*web page)|(?=.*address)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 387,
	name: "This service tracks which web page referred you to it"
} as Regex;