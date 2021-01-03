import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*as long as)|(?=.*purposes))((?=.*necessary)|(?=.*needed)|(?=.*legally obligated))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 178,
	name: "This service keeps user logs for an undefined period of time"
} as Regex;