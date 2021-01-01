import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*not)|(?=.*intentionally))((?=.*interfere)|(?=.*inhibit))((?=.*enjoy)|(?=.*experience))", "i"),
	caseID: 285,
	name: "Users shall not interfere with another person's enjoyment of the service"
} as Regex;