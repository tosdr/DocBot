import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*username)(?=.*user name)((?=.*refuse)|(?=.*reject)|(?=.*remove)|(?=.*disable)|(?=.*cancel)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 328,
	name: "Usernames can be rejected for any reason"
} as Regex;