import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*warrant)|(?=.*guarantee))((?=.*uninterrupted)|(?=.*error-free)|(?=.*error free)|(?=.*secure)|(?=.*timely)|(?=.*timeliness)|(?=.*security))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 287,
	name: "The service provider makes no warranty regarding uninterrupted, timely, secure or error-free service"
} as Regex;