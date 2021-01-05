import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*username)|(?=.*user name)|(?=.*user ID))((?=.refuse)|(?=.reject)|(?=.remove)|(?=.disable)|(?=.cancel)|(?=.change))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 328,
	name: "Usernames can be rejected for any reason"
} as Regex;