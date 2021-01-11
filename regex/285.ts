import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*interfere)|(?=.*inhibit)|(?=.*disrupt)|(?=.*restrict))((?=.*enjoy)|(?=.*experience))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 285,
	name: "Users shall not interfere with another person's enjoyment of the service"
} as Regex;